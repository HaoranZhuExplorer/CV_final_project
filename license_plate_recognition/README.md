# License Plate Recognition

## Introduction

The task is to segment the characters from an image of the rear of a car, then predict the class of the single characters.

The root path is `license_plate_recognition/`.

## Preparation

`Python>=3.8`

## Get started

### Install Miniforge3

> Follow the instructions in https://github.com/conda-forge/miniforge

### Conda virtual environment is **recommended**.

```
conda create -n env python=3.8 -y
conda activate env
```

### Install packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Data preparation

> Dataset for segmentation can be downloaded from https://www.kaggle.com/mobassir/fifty-states-car-license-plates-dataset/version/1 \
> Dataset for OCR training can be downloaded from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/#download

When the downloading is finished, upzip these datasets. Move the folders `Fifty States License Plates` and `English` into the root path. Run the following shell scripts.

```bash
mkdir ocr/EnglishImg
mv English/Img/GoodImg/Bmp ocr/EnglishImg/Train
```

### Run the code

Type in the following shell command line to open the jupyter notebook GUI in a browser.

```bash
jupyter notebook
```

Then you should see at least the data files we just fetched and placed, the `ocr/` folder and the jupyter notebook file `license_plate_recognition.ipynb`.

Open the `.ipynb` file and execute block by block in the interactive interface to test the code and see the results.

For the OCR, to train a new model, please check and run `ocr/Tuning.ipynb`. To load the existing model that I have trained, please check and run `ocr/Predict.ipynb`.
You should be able to find the following lines to change the training parameters and to select the corresponding model.

```python
...

load_batch = 32
load_epochs = 16

...
```
