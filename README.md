
# Arabic to English Translation using Seq2Seq LSTM Model

This project implements a machine translation model for translating Arabic to English using a Seq2Seq architecture with LSTM in PyTorch. The model is trained on the Tatoeba dataset and evaluated using BLEU scores.

## Features
- Seq2Seq architecture with LSTM for machine translation.
- Trained on the Tatoeba dataset (Arabic to English).
- Includes data preprocessing, vocabulary construction, and tokenization.
- Evaluation using BLEU score for translation quality.


### Requirements

You will need the following Python libraries:

- torch (PyTorch)
- numpy
- tqdm
- datasets
- matplotlib
- nltk

## Dataset

The Tatoeba dataset (Arabic to English) is used for training and evaluation. You can access it from Hugging Face's datasets library:

[Hugging Face Tatoeba Dataset](https://huggingface.co/datasets/tatoeba)

To download the dataset, run the following code in your Jupyter notebook:

```python
from datasets import load_dataset

# Load the Tatoeba dataset (Arabic to English)
dataset = load_dataset("tatoeba", lang1="ar", lang2="en")

## Model Description

The model uses a Seq2Seq architecture where:
- **Encoder:** Encodes the input (Arabic sentence) into a hidden state vector.
- **Decoder:** Decodes the hidden state into an output (English translation) using the LSTM model.
- **Seq2SeqWrapper:** A wrapper around the encoder and decoder that manages the overall sequence-to-sequence process, including teacher forcing during training.

The model is trained using CrossEntropy loss and evaluates using BLEU score.
