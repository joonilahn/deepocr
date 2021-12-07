# optimizer config
optimizer = dict(type="Adam", lr=1e-3, weight_decay=1e-5)
optimizer_config = dict(grad_clip=None)

# learning schedule
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=2000,
    warmup_ratio=0.0005,
    step=[50, 100],
)
total_epochs = 120
