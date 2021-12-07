import os
import re
from collections import OrderedDict

import numpy as np
from mmcv.runner import Hook
from mmcv.runner.hooks import CheckpointHook
from torch.utils.data import DataLoader

from ...apis import multi_gpu_test, single_gpu_test
from ...datasets.dataset_wrappers import ConcatDataset
from ...runner import EpochBasedRunner, IterBasedRunner


class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(
        self,
        dataloader,
        interval=1,
        return_acc=True,
        max_num=None,
        save_best=None,
        **eval_kwargs,
    ):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                "dataloader must be a pytorch DataLoader, but got"
                f" {type(dataloader)}"
            )
        self.dataloader = dataloader
        self.interval = interval
        self.return_acc = return_acc
        self.max_num = max_num
        self.save_best = save_best
        self.eval_kwargs = eval_kwargs
        self.eval_history = EvalHistoryLogger()

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

        results = single_gpu_test(
            runner.model,
            self.dataloader,
            return_acc=self.return_acc,
            max_num=self.max_num,
        )
        self.evaluate(runner, results)

    def after_train_iter(self, runner):
        if isinstance(runner, EpochBasedRunner):
            return

        if not self.every_n_iters(runner, self.interval):
            return

        results = single_gpu_test(
            runner.model,
            self.dataloader,
            return_acc=self.return_acc,
            max_num=self.max_num,
        )
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        if isinstance(self.dataloader.dataset, ConcatDataset):
            evaluate_fn = self.dataloader.dataset.datasets[0].evaluate
        else:
            evaluate_fn = self.dataloader.dataset.evaluate
        eval_res = evaluate_fn(results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
            if name == self.save_best:
                if name == "accuracy":
                    val *= 100.0
                self.eval_history.update(val)

        if self.save_best is not None:
            if self.eval_history.is_best_changed:
                for hook in runner.hooks:
                    if isinstance(hook, CheckpointHook):
                        if hook.out_dir is not None:
                            out_dir = hook.out_dir
                        else:
                            out_dir = runner.work_dir
                        break

                if isinstance(runner, EpochBasedRunner):
                    best_weight_name = "best_{}_{:.1f}_epoch_{}.pth".format(
                        self.save_best, self.eval_history.best * 100.0, runner.epoch + 1
                    )
                elif isinstance(runner, IterBasedRunner):
                    best_weight_name = "best_{}_{:.1f}_iter_{}.pth".format(
                        self.save_best, self.eval_history.best * 100.0, runner.iter + 1
                    )
                else:
                    raise NotImplementedError

                runner.save_checkpoint(
                    out_dir,
                    best_weight_name,
                    save_optimizer=False,
                    meta=None,
                    create_symlink=False,
                )
                runner.logger.info(
                    "Best {} score has been changed from {:.1f} to {:.1f}. Saved the new best checkpoint to {}.".format(
                        self.save_best,
                        self.eval_history.prev_best,
                        self.eval_history.best,
                        best_weight_name
                    )
                )

                ckpts = [
                    f
                    for f in os.listdir(out_dir)
                    if f.endswith("pth") and f.startswith("best")
                ]
                for ckpt in ckpts:
                    search_res = re.search(
                        "best_\w+_\d{1,2}\.\d{1}_epoch_\d+\.pth", ckpt
                    )
                    if (search_res is not None) and (ckpt != best_weight_name):
                        os.remove(os.path.join(out_dir, ckpt))

                self.eval_history.is_best_changed = False

        runner.log_buffer.ready = True


class EvalHistoryLogger:
    def __init__(self):
        self._history = []
        self._best_change = False
        self._best = -np.inf
        self._prev_best = -np.inf

    @property
    def history(self):
        return self._history

    @property
    def best_change(self):
        return self._best_change

    @best_change.setter
    def best_change(self, val):
        self._best_change = val

    @property
    def best(self):
        return self._best

    @best.setter
    def best(self, val):
        self._best = val

    @property
    def prev_best(self):
        return self._prev_best

    @prev_best.setter
    def prev_best(self, val):
        self._prev_best = val

    @property
    def is_best_changed(self):
        return self._best_change

    @is_best_changed.setter
    def is_best_changed(self, val):
        self._best_change = val

    def reset(self):
        self._history = []
        self._best_change = False
        self._best = -np.inf
        self._prev_best = -np.inf

    def update(self, val):
        self._history.append(val)
        if val > self._best:
            self._prev_best = self._best
            self._best = val
            self._best_change = True
        else:
            self._best_change = False


class DistEvalHook(EvalHook):
    """Distributed Evaluation hook.
    """

    def __init__(
        self,
        dataloader,
        interval=1,
        return_acc=True,
        max_num=None,
        save_best=None,
        gpu_collect=False,
        **eval_kwargs,
    ):
        super(DistEvalHook, self).__init__(
            dataloader,
            interval=interval,
            return_acc=return_acc,
            max_num=max_num,
            save_best=save_best,
            **eval_kwargs
        )
        self.gpu_collect = gpu_collect

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return

        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            return_acc=self.return_acc,
            tmpdir=os.path.join(runner.work_dir, ".eval_hook"),
            gpu_collect=self.gpu_collect,
        )
        if runner.rank == 0:
            print("\n")
            self.evaluate(runner, results)

    def after_train_iter(self, runner):
        if isinstance(runner, EpochBasedRunner):
            return

        if not self.every_n_iters(runner, self.interval):
            return

        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            return_acc=self.return_acc,
            tmpdir=os.path.join(runner.work_dir, ".eval_hook"),
            gpu_collect=self.gpu_collect,
        )
        if runner.rank == 0:
            print("\n")
            self.evaluate(runner, results)