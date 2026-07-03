import argparse
import json
import logging
import os
import re
import sys
import time
import traceback

import torch
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config.config import config as cfg
from model.TD_draf import DRAFTexDir
from utils import losses
from utils.dataset_sum import DataLoaderX, FaceDatasetFolder
from utils.path_utils import normalize_platform_path
from utils.utils_callbacks import CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

def _now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def _latest_checkpoint_step(output_dir):
    if not os.path.isdir(output_dir):
        return 0
    pattern = re.compile(r"^(\d+)backbone\.pth$")
    best = 0
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match:
            best = max(best, int(match.group(1)))
    return best

def _build_model(local_rank=0, draf_tau: float = 1.0, draf_sparse_lambda: float = 0.001):
    print('Using model definition: model.TD_draf.DRAFTexDir')
    return DRAFTexDir(draf_tau=draf_tau, draf_sparse_lambda=draf_sparse_lambda).to(local_rank)

def _configure_all_data(args):
    cfg.dataset = args.train_dataset
    cfg.output = os.path.join(args.output, cfg.dataset)
    if cfg.dataset == "all_data":
        cfg.rec = args.train_root
        cfg.num_classes = 1722
        cfg.num_image = 26248
    cfg.rec = normalize_platform_path(cfg.rec)

def run_train(args):
    local_rank = 0
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    _configure_all_data(args)
    os.makedirs(cfg.output, exist_ok=True)
    init_logging(logging.getLogger(), local_rank, cfg.output)
    print("traindata_path:", cfg.rec)

    train_status_path = os.path.join(cfg.output, "train_status.json")
    train_done_path = os.path.join(cfg.output, "train_complete.json")
    train_fail_path = os.path.join(cfg.output, "train_failed.json")
    latest_step = _latest_checkpoint_step(cfg.output)

    if os.path.exists(train_done_path) and latest_step > 0:
        logging.info("Training already completed, skip train stage.")
        _write_json(train_status_path, {"status": "completed", "global_step": latest_step, "updated_at": _now_str()})
        return

    if not args.resume and latest_step > 0:
        args.resume = 1
        args.global_step = latest_step
        logging.info("Auto-resume enabled from global step %d", latest_step)

    _remove_if_exists(train_fail_path)
    _write_json(train_status_path, {"status": "running", "global_step": args.global_step, "updated_at": _now_str()})

    try:
        torch.cuda.empty_cache()
        trainset = FaceDatasetFolder(root_dir=cfg.rec, local_rank=local_rank, tm="EBOCV")
        train_loader = DataLoaderX(
            local_rank=local_rank,
            dataset=trainset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            drop_last=True,
            shuffle=True,
        )

        backbone = _build_model(local_rank, args.draf_tau, args.draf_sparse_lambda)
        if args.resume:
            try:
                backbone_pth = os.path.join(cfg.output, f"{args.global_step}backbone.pth")
                backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
                logging.info("backbone resume loaded successfully!")
            except (FileNotFoundError, KeyError, IndexError, RuntimeError):
                logging.info("load backbone resume init, failed!")
        backbone.train()

        header = losses.AdaFace(
            embedding_size=cfg.embedding_size,
            classnum=cfg.num_classes,
            s=cfg.s,
            m=cfg.m,
            h=cfg.h,
            t_alpha=cfg.t_alpha,
        ).to(local_rank)
        if args.resume:
            try:
                header_pth = os.path.join(cfg.output, f"{args.global_step}header.pth")
                header.load_state_dict(torch.load(header_pth, map_location=torch.device(local_rank)))
                logging.info("header resume loaded successfully!")
            except (FileNotFoundError, KeyError, IndexError, RuntimeError):
                logging.info("load header resume init, failed!")
        header.train()

        opt_backbone = torch.optim.SGD(
            params=[{"params": backbone.parameters()}],
            lr=cfg.lr / 512 * args.batch_size,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
        opt_header = torch.optim.SGD(
            params=[{"params": header.parameters()}],
            lr=cfg.lr / 512 * args.batch_size,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
        scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(opt_backbone, lr_lambda=cfg.lr_func)
        scheduler_header = torch.optim.lr_scheduler.LambdaLR(opt_header, lr_lambda=cfg.lr_func)
        criterion = CrossEntropyLoss()

        effective_num_epoch = min(cfg.num_epoch, getattr(cfg, "max_train_epoch", cfg.num_epoch))
        total_step = int(len(trainset) / args.batch_size * effective_num_epoch)
        logging.info("Total Step is: %d", total_step)
        logging.info("Planned Epochs: %d (configured=%d)", effective_num_epoch, cfg.num_epoch)
        logging.info(
            "Runtime config: batch_size=%d, num_workers=%d, pin_memory=%s, draf_sparse_lambda=%.6f, draf_tau=%.4f",
            args.batch_size,
            args.num_workers,
            args.pin_memory,
            args.draf_sparse_lambda,
            args.draf_tau,
        )

        start_epoch = 0
        if args.resume:
            rem_steps = total_step - args.global_step
            cur_epoch = effective_num_epoch - int(effective_num_epoch / total_step * rem_steps)
            start_epoch = cur_epoch
            scheduler_backbone.last_epoch = cur_epoch
            scheduler_header.last_epoch = cur_epoch
            opt_backbone.param_groups[0]["lr"] = scheduler_backbone.get_last_lr()[0]
            opt_header.param_groups[0]["lr"] = scheduler_header.get_last_lr()[0]
            logging.info("resume from estimated epoch %s", cur_epoch)

        callback_logging = CallBackLogging(10, local_rank, total_step, args.batch_size, writer=None)
        callback_checkpoint = CallBackModelCheckpoint(local_rank, cfg.output)
        loss_meter = AverageMeter()
        global_step = args.global_step
        best_epoch_loss = float("inf")
        no_improve_epochs = 0
        converged_epoch = None
        checkpoint_saved = False
        final_epoch = start_epoch

        for epoch in range(start_epoch, effective_num_epoch):
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            epoch_sparse_sum = 0.0
            for _, (img, label, direction) in enumerate(train_loader):
                global_step += 1
                img = img.cuda(local_rank, non_blocking=True)
                label = label.cuda(local_rank, non_blocking=True)
                direction = direction.cuda(local_rank, non_blocking=True)

                features = backbone(img, direction)
                norm = torch.norm(features, 2, 1, True)
                features = torch.div(features, norm)
                if cfg.loss == "AdaFace":
                    thetas = header(features, norm, label)
                else:
                    thetas = header(features, label)

                ce_loss = criterion(thetas, label)
                sparse_loss = backbone.get_aux_loss()
                loss_v = ce_loss + sparse_loss
                loss_v.backward()
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                opt_backbone.step()
                opt_header.step()
                opt_backbone.zero_grad()
                opt_header.zero_grad()
                loss_meter.update(loss_v.item(), 1)
                epoch_loss_sum += loss_v.item()
                epoch_sparse_sum += float(sparse_loss.detach().item())
                epoch_loss_count += 1
                callback_logging(global_step, loss_meter, epoch)
                del features, norm, thetas, loss_v, ce_loss, sparse_loss, img, label, direction

            scheduler_backbone.step()
            scheduler_header.step()
            torch.cuda.empty_cache()

            epoch_avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)
            epoch_avg_sparse = epoch_sparse_sum / max(epoch_loss_count, 1)
            epoch_display = epoch + 1
            final_epoch = epoch_display
            mask_mean = backbone.get_draf_mask_mean()
            mask_mean_value = None if mask_mean is None else float(mask_mean.item())
            logging.info(
                "Epoch %d average loss: %.6f, average sparse loss: %.6f, draf_mask_mean: %s",
                epoch_display,
                epoch_avg_loss,
                epoch_avg_sparse,
                "None" if mask_mean_value is None else f"{mask_mean_value:.6f}",
            )

            improve_threshold = getattr(cfg, "early_stop_min_delta", 5e-5)
            if epoch_avg_loss < (best_epoch_loss - improve_threshold):
                best_epoch_loss = epoch_avg_loss
                no_improve_epochs = 0
                logging.info("Epoch %d improved best loss to %.6f", epoch_display, best_epoch_loss)
            else:
                no_improve_epochs += 1
                logging.info(
                    "Epoch %d shows limited loss improvement (patience %d/%d)",
                    epoch_display,
                    no_improve_epochs,
                    getattr(cfg, "early_stop_patience", 2),
                )

            if epoch_display >= getattr(cfg, "dynamic_save_start_epoch", 6):
                callback_checkpoint(global_step, backbone, header)
                checkpoint_saved = True

            _write_json(
                train_status_path,
                {
                    "status": "running",
                    "epoch": epoch_display,
                    "global_step": global_step,
                    "best_epoch_loss": best_epoch_loss,
                    "avg_sparse_loss": epoch_avg_sparse,
                    "draf_mask_mean": mask_mean_value,
                    "updated_at": _now_str(),
                },
            )

            if (
                converged_epoch is None
                and epoch_display >= getattr(cfg, "early_stop_min_epoch", 8)
                and no_improve_epochs >= getattr(cfg, "early_stop_patience", 2)
            ):
                converged_epoch = epoch_display
                logging.info(
                    "Loss convergence detected at epoch %d. Will continue for %d extra epochs before stopping.",
                    converged_epoch,
                    getattr(cfg, "early_stop_extra_epochs", 3),
                )

            if (
                converged_epoch is not None
                and epoch_display >= converged_epoch + getattr(cfg, "early_stop_extra_epochs", 3)
            ):
                if not checkpoint_saved or epoch_display < getattr(cfg, "dynamic_save_start_epoch", 6):
                    callback_checkpoint(global_step, backbone, header)
                logging.info("Early stopping triggered at epoch %d after convergence tracking.", epoch_display)
                break
        else:
            if not checkpoint_saved:
                callback_checkpoint(global_step, backbone, header)

        final_step = max(global_step, _latest_checkpoint_step(cfg.output))
        torch.cuda.empty_cache()
        done_payload = {"status": "completed", "epoch": final_epoch, "global_step": final_step, "updated_at": _now_str()}
        _write_json(train_done_path, done_payload)
        _write_json(train_status_path, done_payload)
    except Exception as exc:
        torch.cuda.empty_cache()
        _write_json(
            train_fail_path,
            {"status": "failed", "global_step": args.global_step, "error": str(exc), "traceback": traceback.format_exc(), "updated_at": _now_str()},
        )
        _write_json(train_status_path, {"status": "failed", "global_step": args.global_step, "error": str(exc), "updated_at": _now_str()})
        raise


def _parse_args():
    parser = argparse.ArgumentParser(description='Train CMCL-BE DRAF model')
    parser.add_argument('--output', type=str, default=os.path.join(PROJECT_ROOT, 'output'))
    parser.add_argument('--train-dataset', type=str, default='all_data')
    parser.add_argument('--train-root', type=str, default='/mnt/d/lyj/dataset/vein/train_2')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pin-memory', action='store_true', default=True)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--global-step', type=int, default=0)
    parser.add_argument('--draf-sparse-lambda', type=float, default=0.001)
    parser.add_argument('--draf-tau', type=float, default=1.0)
    return parser.parse_args()


if __name__ == '__main__':
    run_train(_parse_args())
