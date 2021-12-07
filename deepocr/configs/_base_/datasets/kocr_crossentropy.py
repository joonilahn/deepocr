# fmt: off
# set path for the dataset
dataset_type = "KOCRDataset"
data_root = "./dataset/kocr/crop/"
corpus = data_root + "corpus.csv"

# define img preprocessing pipeline
img_norm_cfg = dict(mean=[0.209,], std=[0.348,], to_rgb=False)
train_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="LoadAnnotations"),
    dict(type="MaxHeightResize", max_height=32, max_width=288),
    dict(type="RandomRotate", limit=10, border_mode="reflect", p=0.5),
    dict(type="OneOfCorrupt",
         corruptions=[
             dict(name="motion_blur", min=1, max=1),
             dict(name="jpeg_compression", min=1, max=2),
             dict(name="brightness", min=1, max=3),
         ],
         p=0.5,
    ),
    dict(type='InvertColor', zero_to_one_scale=True),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=(32, 288)),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_label"]),
]
val_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="LoadAnnotations"),
    dict(type="MaxHeightResize", max_height=32, max_width=288),
    dict(type='InvertColor', zero_to_one_scale=True),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size=(32, 288)),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
inference_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(
        type="SimpleTestAug",
        transforms=[
            dict(type="MaxHeightResize", max_height=32, max_width=288),
            dict(type='InvertColor', zero_to_one_scale=True),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size=(32, 288)),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=0,
    val_samples_per_gpu=128,
    train=[
        dict(
            type=dataset_type,
            data_root=data_root,
            img_prefix="train.py",
            ann_file="labels_train.csv",
            pipeline=train_pipeline,
            converter=dict(
                type="Seq2SeqConverter",
                 corpus=corpus,
                 max_decoding_length=35,
                 ),
        ),
    ],
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_prefix="val",
        ann_file="labels_val.csv",
        pipeline=val_pipeline,
        converter=dict(
                type="Seq2SeqConverter",
                 corpus=corpus,
                 max_decoding_length=35,
                ),
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_prefix="test",
        ann_file="labels_test.csv",
        pipeline=val_pipeline,
        converter=dict(
                type="Seq2SeqConverter",
                 corpus=corpus,
                 max_decoding_length=35,
                ),
    ),
)

# evaluation config
evaluation = dict(interval=1, output_type='crossentropy', num_show=10, max_num=None, save_best="accuracy")
