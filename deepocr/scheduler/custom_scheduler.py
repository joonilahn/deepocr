from mmcv.runner.hooks import LrUpdaterHook

from .builder import HOOKS


@HOOKS.register_module()
class TransformerLrUpdaterHook(LrUpdaterHook):
    """
    Increase the learning rate linearly for warmup iterations,
    and decrease it thereafter proportionally to the inverse square root
    of the step number.

    Args:
        d_model (int): Embedding dimension of the transformer model.
        warmup_iters (int): Warmup steps for the training.
    """

    def __init__(self, d_model, **kwargs):
        super(TransformerLrUpdaterHook, self).__init__(**kwargs)
        self.d_model = d_model

    def get_lr(self, runner, base_lr):
        if self.by_epoch:
            progress = runner.epoch
        else:
            progress = runner.iter

        # fmt: off
        return self.d_model ** (-0.5) * \
                min(
                    (progress + 1) ** (-0.5),
                    (progress + 1) * self.warmup_iters ** (-1.5),\
                )
        # fmt: on
