# fmt: off
# set path for the dataset
HOMEDIR = "./"
dataset_type = "KOCRDataset"
data_root = HOMEDIR + "dataset/car/mileage/crop/"
corpus =data_root + "corpus.csv"

# define img preprocessing pipeline
img_norm_cfg = dict(mean=[128.5,], std=[67.5,], to_rgb=False)
train_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="LoadAnnotations"),
    dict(type="MaxHeightResize", max_height=32, max_width=100),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=(32, 100)),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_label"]),
]
val_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="LoadAnnotations"),
    dict(type="MaxHeightResize", max_height=32, max_width=100),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=(32, 100)),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
inference_pipeline=[
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(
        type="SimpleTestAug",
        transforms=[
            dict(type="MaxHeightResize", max_height=32, max_width=100),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size=(32, 100)),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=0,
    val_samples_per_gpu=128,
    train=dict(
            type=dataset_type,
            data_root=data_root,
            img_prefix="train.py",
            ann_file="train_label_num.txt",
            pipeline=train_pipeline,
            converter=dict(
                type="CTCConverter",
                 corpus=corpus,
                 max_decoding_length=15,
                 ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_prefix="val",
        ann_file="val_label_num.txt",
        pipeline=val_pipeline,
        converter=dict(
                type="CTCConverter",
                 corpus=corpus,
                 max_decoding_length=15,
                ),
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_prefix="val",
        ann_file="val_label_num.txt",
        pipeline=val_pipeline,
        converter=dict(
                type="CTCConverter",
                 corpus=corpus,
                 max_decoding_length=15,
                ),
    ),
)

# evaluation config
evaluation = dict(interval=1, output_type='ctc', num_show=10, max_num=None, save_best="accuracy")
