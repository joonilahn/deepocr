# checkpoint and log schedule
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=100,
    hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")],
)
log_level = "INFO"

# distributed learning and multi-gpu settings
dist_params = dict(backend="nccl")

# set workflow and work_dir
workflow = [("train", 1)]

# if resuming from a checkpoint
resume_from = None

# if load weights from a pretrained model
load_from = None
