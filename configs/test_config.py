# fmt: off
# set path for the dataset
dataset_type = "KOCRDataset"
data_root = "./data/"
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
    samples_per_gpu=64,
    workers_per_gpu=0,
    val_samples_per_gpu=64,
    train=dict(
            type=dataset_type,
            data_root=data_root,
            img_prefix="train",
            ann_file="sroie_words_train.txt",
            pipeline=train_pipeline,
            converter=dict(
                type="Seq2SeqConverter",
                 corpus=corpus,
                 max_decoding_length=35,
                 ),
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_prefix="test",
        ann_file="sroie_words_test.txt",
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
        ann_file="sroie_words_test.txt",
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

# define model
model = dict(
    type="EncoderDecoder",
    pretrained=None,
    backbone=dict(
        type="ResNetAster",
        in_channels=1,
    ),
    pretransform=dict(
        type="ASTERTransform",
        in_channels=1,
        output_image_size=(32, 288),
        num_control_points=20,
    ),
    encoder=dict(type="LSTMEncoder", input_size=512, hidden_size=256, num_layers=2),
    decoder=dict(
        type="AttentionBidirectionalDecoder",
        input_size=256,
        hidden_size=256,
        num_classes=135,
        loss=dict(type="CrossEntropyLoss", ignore_index=0),
        max_decoding_length=35,
    ),
)

# optimizer config
optimizer = dict(type="Adam", lr=1e-3, weight_decay=1e-5)
optimizer_config = dict(grad_clip=None)

# learning schedule
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=0.0005,
    step=[12,18],
)
total_epochs = 20

# checkpoint and log schedule
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
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

# set work_dir
work_dir = "./work_dir/sroie_test"
