# fmt: off
_base_ = [
    "../_base_/models/crnn_ctc.py",
    "../_base_/datasets/kocr_ctc.py",
    "../_base_/default_runtime.py",
]

# optimizer config
optimizer = dict(type="RAdam", lr=1e-4, weight_decay=5e-4)
optimizer_config = dict(grad_clip=None)

# learning schedule
lr_config = dict(
    policy="step",
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[20, 30],
)
total_epochs = 50

# set work_dir
work_dir = "./kocr_recognizer/work_dir/kocr_crnn_ctc"
