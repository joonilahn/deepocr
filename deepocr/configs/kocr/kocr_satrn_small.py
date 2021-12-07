# fmt: off
_base_ = [
    "../_base_/models/satrn_small.py",
    "../_base_/datasets/kocr_crossentropy.py"
]

# set path for the dataset
dataset_type = "KOCRDataset"
data_root = "./dataset/kocr/crop/"
corpus = data_root + "corpus.csv"

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    val_samples_per_gpu=32,
)

# evaluation config
evaluation = dict(interval=1, output_type='crossentropy', num_show=10, max_num=None, save_best="accuracy")

# optimizer config
optimizer = dict(type="Adam", lr=3e-4, weight_decay=1e-5)
optimizer_config = dict(grad_clip=dict(max_norm=2.0, norm_type=2))

# learning schedule
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=2000,
    warmup_ratio=0.001,
    steps=[12, 40],
)
total_epochs = 50

# log config
log_config = dict(
    interval=10,
    hooks=[dict(type="TextLoggerHook"),],
)

work_dir = "./kocr_recognizer/work_dir/kocr_satrn_small"
