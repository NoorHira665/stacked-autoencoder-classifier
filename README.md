# stacked-autoencoder-classifier

An image classifier built with three stacked convolutional autoencoders trained sequentially in PyTorch. Pre-trained encoder weights are frozen and reused as a feature extractor for downstream classification via transfer learning.

## How It Works

**Step 1: Train three autoencoders in sequence**

Each autoencoder learns to compress and reconstruct images at a progressively smaller spatial resolution:

| Model | Input | Output |
|---|---|---|
| AE1 | 64x64 image | 32x32 encoded |
| AE2 | AE1's encoded output | 16x16 encoded |
| AE3 | AE2's encoded output | 8x8 encoded |

Training each autoencoder to reconstruct its input forces it to learn meaningful visual features rather than memorizing pixel values.

**Step 2: Build a classifier on top**

The three pre-trained encoders are stacked and their weights are frozen. Fully connected layers are added on top to predict one of 4 image classes. This is transfer learning — the encoders already learned useful features, so the classifier just learns how to use them.

## Project Structure

```
stacked-autoencoder-classifier/
├── ae1.py            # First autoencoder (64x64 -> 32x32)
├── ae2.py            # Second autoencoder (32x32 -> 16x16)
├── ae3.py            # Third autoencoder (16x16 -> 8x8)
├── autoencoder.py    # Base autoencoder class
├── cl1.py            # Classifier using stacked encoders
├── classifier.py     # Base classifier class
├── data.py           # Data loading, batching, and display
├── model.py          # Model wrapper with save/load/freeze utilities
├── sample_ae.py      # Baseline autoencoder for comparison
└── sample_cl.py      # Baseline classifier for comparison
```

## Setup

```bash
pip install torch torchvision matplotlib numpy torchinfo
```

Your data folder should be structured as:

```
data/
├── class1/
├── class2/
├── class3/
└── class4/
```

## Usage

Train the autoencoders in order, then train the classifier:

```bash
python ae1.py 20
python ae2.py 20
python ae3.py 20
python cl1.py 20
```

The number is the epoch count. To load a saved model without retraining, run without an argument:

```bash
python cl1.py
```

To run the baseline models for comparison:

```bash
python sample_ae.py 20
python sample_cl.py 20
```

## Tech Stack

`Python` `PyTorch` `torchvision` `torchinfo` `matplotlib` `numpy`
