# AI Homework 7.1 - Attack MNIST Recognition

This project is for Assignment #7.1: Attack MNIST Recognition.

The goal of this assignment is to attack a trained MNIST recognition model and try to fool it using adversarial examples.

I used a CNN model as the target model because it performed the best in my MNIST recognition experiment.

## Dataset

The dataset is MNIST. It contains handwritten digit images from 0 to 9.

The model was trained on the MNIST training set and tested on the MNIST test set.

## Model

The target model is a CNN.

The CNN first uses convolution and pooling layers to extract visual features from the digit image. Then it uses fully connected layers to classify the image into one of 10 classes.

## Attack Methods

I tested three adversarial attack methods:

* FGSM
* PGD / Iterative FGSM
* Momentum I-FGSM

FGSM is a one-step gradient-based attack.

PGD is an iterative attack. It applies multiple small attack steps and keeps the adversarial image inside the allowed perturbation range.

Momentum I-FGSM is similar to iterative FGSM, but it uses momentum to make the attack direction more stable.

## Files

src/models.py      model definition
src/train.py       train the CNN model
src/attack.py      run adversarial attacks
results/           attack results
models/            saved trained model

## How to Run

Install dependencies:


python3 -m pip install -r requirements.txt


Train the CNN model:


python3 src/train.py --model cnn --epochs 3


Run attacks:


python3 src/attack.py --model cnn --epsilon 0.30 --alpha 0.01 --steps 40


## Metrics

I used three main metrics:

* Clean accuracy: model accuracy before attack
* Adversarial accuracy: model accuracy after attack
* Attack success rate, or ASR: how often the attack successfully fools the model

## Results

The clean accuracy before attack was 98.81%.

| Attack          | Clean Accuracy | Adversarial Accuracy | Attack Success Rate |
| --------------- | -------------: | -------------------: | ------------------: |
| FGSM            |         98.81% |                7.85% |              92.06% |
| PGD / I-FGSM    |         98.81% |                0.00% |             100.00% |
| Momentum I-FGSM |         98.81% |                0.00% |             100.00% |

## What I Learned

From this experiment, I learned that a model can have very high accuracy on normal test data but still be weak against adversarial attacks.

FGSM already reduced the accuracy from 98.81% to 7.85%. PGD and Momentum I-FGSM were even stronger and reduced the adversarial accuracy to 0.00%.

This shows that clean accuracy and robustness are different. A model can recognize normal MNIST digits very well, but small gradient-based perturbations can still completely fool it.
