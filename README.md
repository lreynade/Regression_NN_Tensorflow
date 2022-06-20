# Regression_NN_Tensorflow

This is the first project I do using Tensorflow.keras.

This a CodeAcademy project. The goal is to create a deep learning regression model that predicts the
likelihood that a student applying to graduate school will be accepted based 
on various application factors.

The data set has 500 data points so a relatively small number of neurons was used.
The network has two hidden layers. The first one has 64 neurons and the second one 8. The activation
functions are relu and sofmax, respectively.

## Parameters

1. test_size = 0.25
2. learning_rate = 0.001
3. patience = 20
4. Trainable parameters = 1,041
5. epochs = 100
6. batch_size = 3
7. validation_split = 0.12

## Remarks

* Training stops after 53 epochs
* Training: loss = 0.0025 - mae = 0.0363 
* Validation: val_loss = 0.0030 - val_mae = 0.0388
* Test resutls: MSE = 0.0040, MAE = 0.0432
* The r score is 0.8012

The model is overfitting. After tuning some of the parameters similar r scores are obtained. More parameter tuning is needed
to obtain a greater r score or improve the generalization capabilities of the model.
