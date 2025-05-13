
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


## Model Description

The model uses a Seq2Seq architecture where:
- **Encoder:** Encodes the input (Arabic sentence) into a hidden state vector.
- **Decoder:** Decodes the hidden state into an output (English translation) using the LSTM model.

The model is trained using CrossEntropy loss and evaluates using BLEU score.
