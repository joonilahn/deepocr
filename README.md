# DeepOCR implementations in PyTorch

## Introduction
DeepOCR is a ocr framework which provides many types of PyTorch implementations of Text Recognition models.

You can train your own text recognition models using built-in config files that define all the hyperparameters for both models and train setups.

The modular structures and config schemes were influenced by [OpenMMLab's MMDetetion framework](https://github.com/open-mmlab/mmdetection), open source object detection toolbox.

## Supported Models

Pretransform:
- [x] ASTERTransform (STNHead + TPS)

Backbones:
- [x] ResNet
- [x] VGG
- [x] SATRN Backbone

Encoders:
- [x] LSTMEncoder
- [x] TransformerEncoder

Decoders:
- [x] CTCDecoder
- [x] AttentionDecoder1D
- [x] AttentionDecoder2D
- [x] AttentionBidirectionalDecoder
- [x] TransformerDecoder1D
- [x] TransformerDecoder2D

PositionalEncoders:
- [x] PositionalEncoder (1D)
- [x] AdaptivePositionalEncoder2D

Some popular text recognition models, such as ASTER, CRNN-CTC, can be implemented by selecting proper backbones, encoders, and decoders.

For example, ASTER model is a mixture of `ASTERTransform`, `ResNet`, `LSTMEncoder`, and `AttentionBidirectionalDecoder`.

## Getting Started
### Dependency
- This framework was tested with PyTorch 1.4.0, CUDA 10.1, python 3.6 and Ubuntu 18.04.
- But I belive it will be okay if you intalled PyTorch 1.2+, and CUDA 10.0+.

### How to train your own model
1. Download ICDAR dataset.
2. Add image files to train into `dataset/icdar/train`
3. Make annotation files. See `dataset/icdar/train_label.txt` as an example.
4. Run train.py

```
python tools/test.py {config file path} {weight file path}
```

### How to inference

```
python tools/inference.py {config file path} {weight file path} --img {image file path}
```