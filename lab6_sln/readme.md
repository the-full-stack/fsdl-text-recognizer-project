# Lab 6: Line Detection



## Looking at the data

## Data processing

## Network description

The network is defined in `text_recognizer/networks/fcn.py`.
The basic idea is a deep convolutional network with resnet-style blocks (input to block is concatenated to block output).
We call it FCN, as in "Fully Convolutional Network," after the seminal paper that first used convnets for segmentation.

Unlike the original FCN, however, we do not maxpool or upsample, but instead rely on dilated convolutions to rapidly increase the effective receptive field.
With `padding='SAME'`, stacking conv layers results in an output that is exactly the same size as the image, which is what we want.
[Here](https://fomoro.com/projects/project/receptive-field-calculator) is a very calculator of the effective receptive field size of a convnet.

The crucial thing to understand is that because we are labeling odd and even lines differently, each predicted pixel must have the context of the entire image to correctly label -- otherwise, there is no way to know whether the pixel is on an odd or even line.
For this reason, getting to a very high receptive field is crucial.
And because low-level features are still important for the exact extent of the line region, the residual connections are crucial in this network.

## Data augmentation

## Review training results

## Combining the two models
