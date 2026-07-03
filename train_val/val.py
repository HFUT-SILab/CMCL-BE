import argparse
import json
import os
import re
import sys
import time
import traceback

import torch
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from eval import metrics3
from model.TD_draf import DRAFTexDir
from utils.dataset_sum import DataLoader, valDatasetFolder
from utils.path_utils import normalize_platform_path

DATASETS = ['CASIA', 'HFUT', 'PolyU', 'TongJi', 'VERA']

def _now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _remove_if_exists(path):
    if os.path.exists(path):
        os.remove(path)




def _build_model(local_rank=0, draf_tau: float = 1.0, draf_sparse_lambda: float = 0.001):
    print('Using model definition: model.TD_draf.DRAFTexDir')
    return DRAFTexDir(draf_tau=draf_tau, draf_sparse_lambda=draf_sparse_lambda).to(local_rank)

@torch.no_grad()
def _extract_features_with_label(model, dataloader, device=None):
    features = {}
    iterator = tqdm(dataloader)
    for image, label, hog in iterator:
        labels = [l.item() for l in label]
        if device is None:
            image = image.cuda()
            hog = hog.cuda()
        else:
            image = image.to(device, non_blocking=True)
            hog = hog.to(device, non_blocking=True)
        f = model(image, hog)
        for i, target in enumerate(labels):
            features.setdefault(target, []).append(f[i][None])
    ordered = []
    for key in sorted(features.keys()):
        value = torch.cat(features[key], dim=0)
        value = torch.nn.functional.normalize(value, p=2, dim=1)
        ordered.append(value.cpu().numpy())
    return ordered

def run_val(args):
    local_rank = 0
    torch.cuda.set_device(local_rank)
    pth_path = os.path.join(args.output, f"{args.train_split}_data")
    val_status_path = os.path.join(pth_path, "val_status.json")
    val_done_path = os.path.join(pth_path, "val_complete.json")
    val_fail_path = os.path.join(pth_path, "val_failed.json")
    val_root = normalize_platform_path(args.val_root)

    os.makedirs(pth_path, exist_ok=True)
    if os.path.exists(val_done_path):
        print("Validation already completed, skip val stage.")
        _write_json(val_status_path, {"status": "completed", "updated_at": _now_str()})
        return

    _remove_if_exists(val_fail_path)
    _write_json(val_status_path, {"status": "running", "updated_at": _now_str()})

    try:
        torch.cuda.empty_cache()
        checkpoint_files = (
            sorted(
                [name for name in os.listdir(pth_path) if name.endswith("backbone.pth")],
                key=lambda name: int(name.split("backbone")[0]),
            )
            if os.path.exists(pth_path)
            else []
        )

        if not checkpoint_files:
            print(f"skip validation, checkpoint path not found or empty: {pth_path}")
            _write_json(val_status_path, {"status": "waiting_for_checkpoint", "updated_at": _now_str()})
            return

        for dataset_name in args.datasets:
            txt_path = os.path.join(args.output, dataset_name)
            os.makedirs(txt_path, exist_ok=True)
            print(f"{args.train_split} train {dataset_name} start!")

            for p in checkpoint_files:
                num = int(p.split("backbone")[0])
                result_prefix = os.path.join(txt_path, f"{args.train_split}_train{num}_{dataset_name}")
                result_file = f"{result_prefix}_results.txt"
                if os.path.exists(result_file):
                    print(f"skip existing result: {result_file}")
                    continue

                print("pth", num, ":start test!")
                test_dataset = valDatasetFolder(root_dir=os.path.join(val_root, dataset_name), local_rank=local_rank, tm="EBOCV")
                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=args.pin_memory,
                    drop_last=False,
                )

                model = _build_model(local_rank, args.draf_tau, args.draf_sparse_lambda)
                model.load_state_dict(torch.load(os.path.join(pth_path, p), map_location="cuda:0"))
                model = model.cuda()
                model.eval()
                features = _extract_features_with_label(model, test_loader)
                metrics3.calculate_eer_multi(features, result_prefix, nproc=args.nproc)
                torch.cuda.empty_cache()
                _write_json(
                    val_status_path,
                    {"status": "running", "dataset": dataset_name, "checkpoint_step": num, "updated_at": _now_str()},
                )

        _write_json(val_done_path, {"status": "completed", "updated_at": _now_str()})
        _write_json(val_status_path, {"status": "completed", "updated_at": _now_str()})
        torch.cuda.empty_cache()
    except Exception as exc:
        torch.cuda.empty_cache()
        _write_json(val_fail_path, {"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "updated_at": _now_str()})
        _write_json(val_status_path, {"status": "failed", "error": str(exc), "updated_at": _now_str()})
        raise


def _parse_args():
    parser = argparse.ArgumentParser(description='Validate CMCL-BE DRAF model checkpoints')
    parser.add_argument('--output', type=str, default=os.path.join(PROJECT_ROOT, 'output'))
    parser.add_argument('--val-root', type=str, default='/mnt/d/lyj/dataset/vein/val')
    parser.add_argument('--train-split', type=str, default='all')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--pin-memory', action='store_true', default=True)
    parser.add_argument('--draf-sparse-lambda', type=float, default=0.001)
    parser.add_argument('--draf-tau', type=float, default=1.0)
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--nproc', type=int, default=4)
    return parser.parse_args()


if __name__ == '__main__':
    run_val(_parse_args())
