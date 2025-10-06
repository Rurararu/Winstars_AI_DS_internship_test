# Task 2: Named entity recognition + image classification

## Description

This part of the project implements a multi-modal ML pipeline that combines Natural Language Processing (NLP) and Computer Vision (CV) for verifying user-provided text descriptions of animals in images. The goal is to determine whether the text correctly identifies the animal present in the image.

## Implemented Models

1. **Text Analysis (NER)** - Transformer-based model for detecting animal names in text
2. **Image Classification (ResNet50)** – Model trained on a dataset of multiple animals (10 classes)
3. **Pipeline Script** - Accepts text and image inputs and returns a boolean output

## Project Structure

```
task2/
├── README.md
├── requirements.txt
├── data/
│   ├── animals/
│   │   ├── dog/
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   └── ...
│   │   ├── cat/
│   │   │   └── ...
│   │   ├── horse/
│   │   │   └── ...
│   │   └── ... (10+ classes)
│   └── ner/
│   │   ├── test.txt
│   │   ├── train.txt
│   │   └── val.txt
├── notebooks/
│   ├── eda.ipynb
│   └── demo.ipynb
├── models/
│   ├── ner/
│   └── image_classification/
├── src/
│   ├── ner_train.py
│   ├── ner_inference.py
│   ├── image_train.py
│   ├── image_inference.py
│   └── pipeline.py                           
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd task2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Used dataset:

**Dataset:** [Animal Image Dataset (90 Different Animals)](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

## Usage Overview

- Prepare or download the animal image dataset and NER text dataset as shown in the `data/` folder.
- Train the **Image Classification model** using `src/image_train.py`.
- Use `src/image_inference.py` for image predictions.
- **NER model** and **pipeline script** are planned but not fully implemented yet.

## Implemented vs Still to Implement

**Implemented:**

- Jupyter notebook with exploratory data analysis of the image dataset (`notebooks/eda.ipynb`)
- Parametrized training and inference `.py` files for the **Image Classification** model (`src/image_train.py`, `src/image_inference.py`)

**Still to be implemented:**

- Parametrized training and inference `.py` files for the **NER** model (`src/ner_train.py`, `src/ner_inference.py`)
- Python script for the entire **pipeline** that takes 2 inputs (text and image) and provides 1 boolean value as output (`src/pipeline.py`)
