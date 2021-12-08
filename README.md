# Deep Learning OCR Framework in PyTorch

## Introduction
DeepOCR is a OCR framework which provides many types of PyTorch implementations for deep learning based Text Recognition models.

You can train your own text recognition models using built-in config files that define all the hyperparameters for both models and train setups.

The modular structures and config schemes were influenced by [OpenMMLab's MMDetetion framework](https://github.com/open-mmlab/mmdetection), open source object detection toolbox.

## Supported Models

Recognizers:
- [x] CRNN-CTC [(https://arxiv.org/abs/1511.04176)](https://arxiv.org/abs/1511.04176)
- [x] ASTER [(https://paperswithcode.com/paper/aster-an-attentional-scene-text-recognizer)](https://paperswithcode.com/paper/aster-an-attentional-scene-text-recognizer)
- [x] SAR (Show, Attend and Read) [(https://arxiv.org/abs/1811.00751)](https://arxiv.org/abs/1811.00751)
- [x] SATRN (On Recognizing Texts of Arbitrart Shapes with 2D Self-Attention) [(https://arxiv.org/abs/1910.04396)](https://arxiv.org/abs/1910.04396)
- [x] SRN (Towards Accurate Scene Text Recognition with Semantic Reasoning Networks) [(https://arxiv.org/abs/2003.12294)](https://arxiv.org/abs/2003.12294)

You can also customize your own text recognizer models with combinations of the following modules.

Pretransform:
- [x] ASTERTransform (STNHead + TPS)

Backbones:
- [x] ResNet
- [x] VGG
- [x] SATRN Backbone

Encoders:
- [x] LSTMEncoder
- [x] TransformerEncoder1D
- [x] TransformerEncoder2D
- [x] PVAM (Parallel Visual Attention Model) 

Decoders:
- [x] CTCDecoder
- [x] AttentionDecoder1D
- [x] AttentionDecoder2D
- [x] AttentionBidirectionalDecoder
- [x] TransformerDecoder1D
- [x] TransformerDecoder2D
- [x] VSFD (Visual-Semantic Fusion Decoder)

PositionalEncoders:
- [x] PositionalEncoder (1D)
- [x] AdaptivePositionalEncoder2D

Some popular text recognition models, such as ASTER, CRNN-CTC, can be implemented by selecting proper backbones, encoders, and decoders.

For example, ASTER model is a mixture of `ASTERTransform`, `ResNet`, `LSTMEncoder`, and `AttentionBidirectionalDecoder`.

## Getting Started
### Dependency
- This framework was tested with PyTorch 1.6.0, CUDA 10.1, python 3.7 and Ubuntu 18.04.

### How to train your own model
1. Download ICDAR dataset.
2. Add image files to train into `dataset/icdar/train` (To be uploaded soon)
3. Make annotation files. See `dataset/icdar/train_label.txt` as an example. (To be uploaded soon)
4. Run train.py

```
python tools/test.py {config file path} {weight file path}
```

### How to inference

```
python tools/inference.py {config file path} {weight file path} --img {image file path}
```