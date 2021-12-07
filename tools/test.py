import argparse

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from deepocr.apis import single_gpu_test
from deepocr.datasets import build_dataloader, build_dataset
from deepocr.datasets.dataset_wrappers import ConcatDataset
from deepocr.models import build_recognizer


def parse_args():
    parser = argparse.ArgumentParser(description='DeepOCR test (and eval) a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='a checkpoint file')
    parser.add_argument(
        "--exclude", type=str, nargs="+", help="string to be excluded in evaluation"
    )
    parser.add_argument("--out", help="result file path")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="number of samples per iteration"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="number of workers for loading data"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print results"
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # build dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False
    )

    # build the model and load checkpoint
    model = build_recognizer(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(
        model, data_loader, return_acc=True, return_filenames=True
    )

    # evaluation
    verbose = True if args.verbose else False
    if isinstance(dataset, ConcatDataset):
        evaluate_fn = dataset.datasets[0].evaluate
    else:
        evaluate_fn = dataset.evaluate
    eval_res = evaluate_fn(
        outputs,
        output_type=cfg.evvaluation.output_type,
        exclude=args.exclude,
        print_all=verbose,
    )

    print(
        "\nNumber of Test Data: {}, Test Accuracy: {:.1f}%".format(
            len(dataset), eval_res["accuracy"] * 100.0
        )
    )

if __name__ == '__main__':
    main()