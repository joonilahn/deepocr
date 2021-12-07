# define model
model = dict(
    type="SAR",
    pretrained=None,
    backbone=dict(type="SARBackbone", in_channels=1,),
    pretransform=None,
    encoder=dict(
        type="LSTMEncoder",
        input_size=512,
        hidden_size=512,
        num_layers=2,
        cell_type="LSTM",
        bidirectional=False,
    ),
    decoder=dict(
        type="AttentionDecoder2D",
        input_size=512,
        hidden_size=512,
        num_classes=2468,
        loss=dict(type="NLLLoss", ignore_index=0),
        max_decoding_length=35,
    ),
)
