
# Arabic to English Translation using Seq2Seq LSTM Model

This project implements a machine translation model for translating Arabic to English using a Seq2Seq architecture with LSTM in PyTorch. The model is trained on the Tatoeba dataset and evaluated using BLEU scores.

## Features
- Seq2Seq architecture with LSTM for machine translation.
- Trained on the Tatoeba dataset (Arabic to English).
- Includes data preprocessing, vocabulary construction, and tokenization.
- Evaluation using BLEU score for translation quality.

## Installation

### Prerequisites

Make sure you have Python 3.6 or higher installed.

To install the required dependencies, you can use `pip`:

```bash
pip install -r requirements.txt
```

### Dataset

To install and download the Tatoeba dataset, run:

```bash
python download_dataset.py
```

This script will download the dataset and prepare it for training.

### Requirements

You will need the following Python libraries:

- torch (PyTorch)
- numpy
- tqdm
- datasets
- matplotlib
- nltk
- scikit-learn

These can be installed via the `requirements.txt` file:

```text
torch>=1.10.0
numpy
tqdm
datasets
matplotlib
nltk
scikit-learn
```

## Usage

### Training

To train the model, simply run the following command:

```bash
python train.py
```

This will start the training process, saving the model checkpoints during each epoch.

### Evaluation

After training, you can evaluate the model using the following command:

```bash
python evaluate.py
```

This will output the evaluation loss and BLEU score.

## Model Description

The model uses a Seq2Seq architecture where:
- **Encoder:** Encodes the input (Arabic sentence) into a hidden state vector.
- **Decoder:** Decodes the hidden state into an output (English translation) using the LSTM model.

The model is trained using CrossEntropy loss and evaluates using BLEU score.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
