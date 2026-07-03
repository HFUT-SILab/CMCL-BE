import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATASETS = ['CASIA', 'HFUT', 'PolyU', 'TongJi', 'VERA']


def _run(command):
    env = os.environ.copy()
    env['PYTHONPATH'] = PROJECT_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    print('\n' + '=' * 80)
    print(' '.join(command))
    print('=' * 80)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)


def _parse_args():
    parser = argparse.ArgumentParser(description='Run CMCL-BE train, validation, and best-result summary')
    parser.add_argument('--output', type=str, default=os.path.join(PROJECT_ROOT, 'output'))
    parser.add_argument('--train-root', type=str, default='/mnt/d/lyj/dataset/vein/train_2')
    parser.add_argument('--val-root', type=str, default='/mnt/d/lyj/dataset/vein/val')
    parser.add_argument('--train-dataset', type=str, default='all_data')
    parser.add_argument('--train-split', type=str, default='all')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--draf-sparse-lambda', type=float, default=0.001)
    parser.add_argument('--draf-tau', type=float, default=1.0)
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-val', action='store_true')
    parser.add_argument('--skip-best', action='store_true')
    return parser.parse_args()


def main():
    args = _parse_args()
    if not args.skip_train:
        _run([
            sys.executable, os.path.join(SCRIPT_DIR, 'train.py'),
            '--output', args.output,
            '--train-dataset', args.train_dataset,
            '--train-root', args.train_root,
            '--batch-size', str(args.batch_size),
            '--num-workers', str(args.num_workers),
            '--draf-sparse-lambda', str(args.draf_sparse_lambda),
            '--draf-tau', str(args.draf_tau),
        ])
    if not args.skip_val:
        _run([
            sys.executable, os.path.join(SCRIPT_DIR, 'val.py'),
            '--output', args.output,
            '--val-root', args.val_root,
            '--train-split', args.train_split,
            '--batch-size', str(args.batch_size),
            '--draf-sparse-lambda', str(args.draf_sparse_lambda),
            '--draf-tau', str(args.draf_tau),
            '--datasets', *args.datasets,
            '--nproc', str(args.nproc),
        ])
    if not args.skip_best:
        _run([sys.executable, os.path.join(SCRIPT_DIR, 'show_best.py'), '--output', args.output, '--train-split', args.train_split])


if __name__ == '__main__':
    main()
