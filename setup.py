#!/usr/bin/env python
from setuptools import find_packages, setup


setup(
    name="deepocr",
    version="1.0",
    author="Joonil Ahn",
    email="joonilahn1@gmail.com",
    description="Deep Learning Based OCR Framework",
    keywords="optical character recognition",
    packages=find_packages(),
    install_requires=[
        "mmcv-full==1.3.1",
        "Pillow",
        "tensorboardX>=2.0",
        "opencv-python>=4.4.0",
        "numpy>=1.18.1",
        "albumentations==0.4.3",
        "imagecorruptions==1.1.0",
        "soynlp",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    include_package_data=True,
    zip_safe=False,
)
