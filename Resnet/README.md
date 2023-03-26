# Resnet Model
ResNet models have contributed significantly to the usage of extremely deep neural
networks; by minimizing the loss of gradient in the deepest levels of the neural network by adding a residual connection between each convolution layer
## Create mel images
```
   bash
   python mel_images
   ```
## prepare the data 
```
   bash
   python prepare_data
   ```
## Train the model

```
   bash
   python train_resnet.py
   
```
## Evaluate the model
In this part we can see the accuracy In the test data and the cnfusion matrix.

```
bash
python test.py
```