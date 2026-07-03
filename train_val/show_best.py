import argparse
import os
import re

DATASETS = ['CASIA', 'HFUT', 'PolyU', 'TongJi', 'VERA']
RESULT_PATTERN = re.compile(r'^(?P<train>.+)_train(?P<epoch>\d+)_(?P<dataset>.+)_results\.txt$')

def _parse_result_file(path):
    metrics = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "-->" not in line:
                continue
            key, value = line.split("-->", 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                continue
    return metrics


def _collect_epochs(output_root, train_name="all"):
    dataset_dir = os.path.join(output_root, DATASETS[0])
    if not os.path.isdir(dataset_dir):
        return []
    epochs = []
    for name in os.listdir(dataset_dir):
        match = RESULT_PATTERN.match(name)
        if match and match.group("train") == train_name:
            epochs.append(int(match.group("epoch")))
    return sorted(set(epochs))


def _dataset_result_path(output_root, dataset, train_name, epoch):
    return os.path.join(output_root, dataset, f"{train_name}_train{epoch}_{dataset}_results.txt")


def _best_average_epoch(output_root, train_name="all", metric_key="TAR_FAR_E6"):
    best_epoch = None
    best_mean = float("-inf")
    best_metrics = {}
    for epoch in _collect_epochs(output_root, train_name):
        current = {}
        values = []
        valid = True
        for dataset in DATASETS:
            path = _dataset_result_path(output_root, dataset, train_name, epoch)
            if not os.path.exists(path):
                valid = False
                break
            metrics = _parse_result_file(path)
            if metric_key not in metrics:
                valid = False
                break
            current[dataset] = metrics
            values.append(metrics[metric_key])
        if valid:
            mean_value = sum(values) / len(values)
            if mean_value > best_mean:
                best_mean = mean_value
                best_epoch = epoch
                best_metrics = current
    return best_epoch, best_mean, best_metrics


def _best_single_epochs(output_root, train_name="all", metric_key="TAR_FAR_E6"):
    results = {}
    epochs = _collect_epochs(output_root, train_name)
    for dataset in DATASETS:
        best_epoch = None
        best_value = float("-inf")
        best_metrics = {}
        for epoch in epochs:
            path = _dataset_result_path(output_root, dataset, train_name, epoch)
            if not os.path.exists(path):
                continue
            metrics = _parse_result_file(path)
            if metric_key in metrics and metrics[metric_key] > best_value:
                best_epoch = epoch
                best_value = metrics[metric_key]
                best_metrics = metrics
        if best_epoch is not None:
            results[dataset] = {"epoch": best_epoch, "metric": best_value, "metrics": best_metrics}
    return results


def _write_metrics_block(handle, dataset_name, epoch, metrics):
    handle.write(f"{dataset_name}\n")
    handle.write(f"epoch --> {epoch}\n")
    for key, value in metrics.items():
        handle.write(f"{key} --> {value}\n")
    handle.write("\n")


def run_best(args):
    output_root = args.output
    train_name = args.train_split
    result_dir = os.path.join(output_root, "result")
    os.makedirs(result_dir, exist_ok=True)

    avg_path = os.path.join(result_dir, "result_e6_avg.txt")
    single_path = os.path.join(result_dir, "result_e6_single_best.txt")

    best_epoch, best_mean, best_metrics = _best_average_epoch(output_root, train_name, "TAR_FAR_E6")
    with open(avg_path, "w", encoding="utf-8") as f:
        if best_epoch is None:
            f.write("No valid result files found.\n")
        else:
            f.write(f"{best_epoch}\n")
            for dataset in DATASETS:
                _write_metrics_block(f, dataset, best_epoch, best_metrics[dataset])
            f.write(f"{best_mean}\n")

    single_results = _best_single_epochs(output_root, train_name, "TAR_FAR_E6")
    with open(single_path, "w", encoding="utf-8") as f:
        if not single_results:
            f.write("No valid result files found.\n")
        else:
            for dataset in DATASETS:
                info = single_results.get(dataset)
                if info is not None:
                    _write_metrics_block(f, dataset, info["epoch"], info["metrics"])

    print(f"Saved average-best report to: {avg_path}")
    print(f"Saved per-dataset-best report to: {single_path}")


def _parse_args():
    parser = argparse.ArgumentParser(description='Summarize best CMCL-BE validation results')
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--train-split', type=str, default='all')
    return parser.parse_args()


if __name__ == '__main__':
    run_best(_parse_args())
