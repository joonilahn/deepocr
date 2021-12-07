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
        num_classes=2468,
        loss=dict(type="CrossEntropyLoss", ignore_index=0),
        max_decoding_length=35,
    ),
)
