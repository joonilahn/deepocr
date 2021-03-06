# fmt:off
model = dict(
    type="SATRN",
    pretrained=None,
    backbone=dict(
        type="SATRNBackbone",
         in_channels=1,
    ),
    src_positional_encoder=dict(
        type="AdaptivePositionalEncoder2D",
        height=8,
        width=72,
        embed_dim=512,
    ),
    tgt_positional_encoder=dict(
        type="PositionalEncoder",
        seq_len=35,
        embed_dim=512,
    ),
    encoder=dict(
        type="TransformerEncoder2D",
        embed_dim=512,
        nhead=8,
        height=8,
        width=72,
        num_layers=9,
        dim_feedforward=2048,
        dropout=0.1,
    ),
    decoder=dict(
        type="TransformerDecoder2D",
        embed_dim=512,
        nhead=8,
        height=8,
        width=72,
        num_layers=3,
        num_classes=2468,
        dim_feedforward=2048,
        dropout=0.1,
        loss=dict(
            type="CrossEntropyLoss", 
            ignore_index=0
        ),
    ),
    pad_id=0,
    num_classes=2468,
)
