# fmt: off
_base_ = [
    "../_base_/models/sar.py",
    "../_base_/datasets/kocr_crossentropy.py",
]

# optimizer config
optimizer = dict(type="Adam", lr=1e-3, weight_decay=1e-5)
optimizer_config = dict(grad_clip=None)

# learning schedule
lr_config = dict(
    policy="ReducOnPlateau",
    threshold=0.001,
    patience=5,
    verbose=True,
    warmup="linear",
    warmup_iters=2000,
    warmup_ratio=0.001,
)
total_epochs = 150

# work_dir
work_dir = "./kocr_recognizer/work_dir/kocr_sar"
