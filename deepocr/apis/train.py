import random

import numpy as np
import torch

from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, OptimizerHook, build_optimizer
from ..core import EvalHook
from ..datasets import build_dataloader, build_dataset
from ..runner import EpochBasedRunner, IterBasedRunner
from ..scheduler import *
from ..utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
    meta=None,
    workers_per_gpu=None,
    runner_type="epoch",
):
    logger = get_root_logger(cfg.log_level)

    # setup workers_per_gpu
    if workers_per_gpu is None:
        workers_per_gpu = cfg.data.workers_per_gpu

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if "imgs_per_gpu" in cfg.data:
        logger.warning(
            '"imgs_per_gpu" is deprecated in MMDet V2.0. '
            'Please use "samples_per_gpu" instead'
        )
        if "samples_per_gpu" in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f"={cfg.data.imgs_per_gpu} is used in this experiments"
            )
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f"{cfg.data.imgs_per_gpu} in this experiments"
            )
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    # dataloaders
    data_loaders = []
    for ds in dataset:
        loader = build_dataloader(
            dataset=ds,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=workers_per_gpu,
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
        )
        data_loaders.append(loader)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    if runner_type == "epoch":
        runner = EpochBasedRunner(
            model, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta
        )
    elif runner_type == "iter":
        runner = IterBasedRunner(
            model, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta
        )
    else:
        raise NotImplementedError("{} is not a proper runner_type.".format(runner_type))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    if distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val)
        val_dataloader = build_dataloader(
            dataset=val_dataset,
            samples_per_gpu=cfg.data.val_samples_per_gpu,
            workers_per_gpu=workers_per_gpu,
            dist=distributed,
            shuffle=True,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_hook = EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    if runner_type == "epoch":
        runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
    elif runner_type == "iter":
        runner.run(data_loaders, cfg.workflow, cfg.total_iters)
    else:
        raise NotImplementedError
