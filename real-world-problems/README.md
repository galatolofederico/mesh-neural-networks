# Real World Problems

This folder contains the experiements for the chapter 4.2 "Real-world problems"

## Installation

Create a virtual environment and activate it

```
virtualenv --python=python3.6 env
. ./env/bin/activate
```

Install the requirements

```
pip install -r requirements.txt
```

## Experiments

Train the Convolutional Autoencoder(CAE) and extract the features

```
python features-extraction.py --dataset mnist --output datasets/mnist
python features-extraction.py --dataset fashion --output datasets/fashion
```

Run the experiments of the paper

MNIST
```
python experiments.py --dataset-folder datasets/mnist --arch mnn --mnn-ticks 2 --mnn-hidden 50 --mnn-zero-prob 0.85
python experiments.py --dataset-folder datasets/mnist --arch mnn --mnn-ticks 2 --mnn-hidden 50 --mnn-zero-prob 0.80
python experiments.py --dataset-folder datasets/mnist --arch mlp --mlp-hidden 13
python experiments.py --dataset-folder datasets/mnist --arch mlp --mlp-hidden 17
```

Fashion MNIST
```
python experiments.py --dataset-folder datasets/fashion --arch mnn --mnn-ticks 2 --mnn-hidden 50 --mnn-zero-prob 0.85
python experiments.py --dataset-folder datasets/fashion --arch mnn --mnn-ticks 2 --mnn-hidden 50 --mnn-zero-prob 0.80
python experiments.py --dataset-folder datasets/fashion --arch mlp --mlp-hidden 13
python experiments.py --dataset-folder datasets/fashion --arch mlp --mlp-hidden 17
```

A fully working **numpy only** implementation is also provided in the file `mnn-numpy.py`

