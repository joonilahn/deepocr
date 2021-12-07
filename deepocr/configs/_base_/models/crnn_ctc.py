# define model
model = dict(
    type="CRNN",
    pretrained=None,
    backbone=dict(type="VGG7", in_channels=1,),
    pretransform=None,
    encoder=None,
    decoder=dict(
        type="CTCDecoder",
        input_size=512,
        hidden_size=256,
        num_classes=2468,
        loss=dict(type="CTCLoss", zero_infinity=True),
    ),
)
