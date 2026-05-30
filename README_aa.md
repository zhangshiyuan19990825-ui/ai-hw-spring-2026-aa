# AI Homework 7.1: Attack MNIST Recognition

## Project Overview

This project is for Assignment 7.1, Attack MNIST Recognition. The goal of this assignment is to attack a trained MNIST recognition model and evaluate how easily the model can be fooled by adversarial examples.

The same MNIST dataset was used. MNIST contains handwritten digit images from 0 to 9. The model was trained on the MNIST training set and evaluated on the MNIST test set.

The CNN model was used as the target model for the attack because it performed the best in Assignment 4.1.

The main files in this project are:

| File            | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `src/models.py` | Defines the CNN target model                                 |
| `src/train.py`  | Trains the CNN model                                         |
| `src/attack.py` | Runs adversarial attacks on the trained CNN model            |
| `results/`      | Stores training history, attack results, and trained weights |

## Target Model

The target model in this assignment is a CNN, which stands for Convolutional Neural Network.

The CNN model first uses convolution and pooling layers to extract visual features from the digit image, such as edges, strokes, and digit shapes. Then it uses fully connected layers to classify the image into one of the 10 digit classes.

The reason for choosing CNN as the target model is that it had the highest test accuracy in Assignment 4.1.

## Target Model Training History

Before running adversarial attacks, the CNN target model was trained for 3 epochs. The clean test accuracy during training is shown below.

| Model | Epoch | Clean Test Accuracy |
| ----- | ----: | ------------------: |
| CNN   |     1 |              98.18% |
| CNN   |     2 |              98.59% |
| CNN   |     3 |              98.81% |

After 3 epochs, the CNN reached 98.81% clean accuracy on the MNIST test set. This shows that the target model performed very well on normal MNIST images before any adversarial attack.

## Attack Methods

Three adversarial attack methods were tested: FGSM, PGD / Iterative FGSM, and Momentum I-FGSM.

| Attack Method   | Description                                                                     |
| --------------- | ------------------------------------------------------------------------------- |
| FGSM            | A one-step gradient-based adversarial attack                                    |
| PGD / I-FGSM    | An iterative attack that applies multiple small gradient-based steps            |
| Momentum I-FGSM | An iterative attack that uses momentum to make the attack direction more stable |

The main idea of these attacks is to use gradient information from the trained model to slightly modify the input image. The change may appear small to humans, but it can cause the model to make an incorrect prediction.

In this experiment, epsilon was set to 0.30. For PGD and Momentum I-FGSM, alpha was set to 0.01 and the number of steps was set to 40.

## Metrics

Three metrics were used to evaluate the attack results.

| Metric                      | Meaning                                                                |
| --------------------------- | ---------------------------------------------------------------------- |
| Clean Accuracy              | Model accuracy before the attack                                       |
| Adversarial Accuracy        | Model accuracy after the attack                                        |
| Attack Success Rate, or ASR | The percentage of cases where the attack successfully fooled the model |

## Attack Results

The attack results are shown below.

| Attack          | Clean Accuracy | Adversarial Accuracy | Attack Success Rate |
| --------------- | -------------: | -------------------: | ------------------: |
| FGSM            |         98.81% |                7.85% |              92.06% |
| PGD / I-FGSM    |         98.81% |                0.00% |             100.00% |
| Momentum I-FGSM |         98.81% |                0.00% |             100.00% |

The FGSM attack reduced the model accuracy from 98.81% to 7.85%, with an attack success rate of 92.06%.

The PGD / I-FGSM attack was stronger. It reduced the adversarial accuracy to 0.00%, with a 100.00% attack success rate.

Momentum I-FGSM also reduced the adversarial accuracy to 0.00%, with a 100.00% attack success rate.

## Weights in the Attack Project

In the attack project, the weights are the learned parameters of the CNN target model. These weights were updated during the training stage through backpropagation and the Adam optimizer.

During the adversarial attack stage, the trained model weights were fixed. The attack did not retrain the model and did not change the model weights. Instead, the attack changed the input images in order to fool the already trained model.

This distinction is important. Training updates model weights, while adversarial attacking modifies input images.

| Weight File | Description                                                          |
| ----------- | -------------------------------------------------------------------- |
| `cnn.pt`    | Trained weights for the CNN target model used in adversarial attacks |

## Assignment 7.1 Conclusion

The attack experiment shows that high accuracy on clean test data does not necessarily mean that a model is robust.

The CNN model achieved 98.81% clean accuracy on normal MNIST test images. However, after adversarial attacks, the performance dropped significantly. FGSM reduced the adversarial accuracy to 7.85%, while PGD and Momentum I-FGSM reduced the adversarial accuracy to 0.00%.

This result shows that a model can classify normal images very well but still fail when the input is carefully modified by an adversarial attack.

Overall, this assignment shows the difference between clean accuracy and robustness. A model may perform well on normal test data, but it can still be vulnerable to adversarial examples.
