<!-- Hello everyone

This is BEEEEEEEtorch :  a pytorch based library to make it easy to train and test machine learning models.

It also contains a part to get notifications (through slack)????? Pushbullet to stay notified when your model stopped training, got the desired accuracy, etc..

Here are the dependencies :

- numpy
- torch, torchmetrics
- mysql-connector-python
- pushbullet.py

ToDo list:
Scikit : SVM and kernel regression + polynomial with a grid search (better accuracy?)
We can't save all models locally in the saves file
Make a polynomial and kernel classes (and use sklearn to implement them)
implement things like convolution in D to increase the dimension (we can use smthg like a sliding window and insert between pixels the sum/product/non-linearity of adjacent matrices (better do it on 2D where adjacency is more relevent then flatten )) -->

# BEEEEEEEtorch

Beetorch is a PyTorch-based library designed to simplify the training and testing of machine learning models. It also includes features for receiving notifications (via Slack or Pushbullet) about your model's progress, such as when training has stopped or a desired accuracy has been achieved.
--------------------------------------------------------------------------------------
- Added randomness to the label flipping


## Features

- Simplified training and testing of machine learning models
- Notifications via Slack or Pushbullet when your model reaches certain milestones

## Dependencies

To use BEEEEEEEtorch, you will need the following Python packages:

- numpy
- torch, torchmetrics
- mysql-connector-python
- pushbullet.py

## Installation

You can install these packages using pip:

```bash
pip install numpy torch torchmetrics mysql-connector-python pushbullet.py
```

## To-Do List

- **Improve Model Accuracy:** Explore the use of Support Vector Machines (SVMs), kernel regression, and polynomial regression with grid search using Scikit-learn to potentially improve model accuracy.
- **Model Saving:** Address the issue of not being able to save all models locally in the 'saves' file.
- **Class Creation:** Create Polynomial and Kernel classes, leveraging Scikit-learn for their implementation.
- **Dimensionality Increase:** Implement techniques such as convolution in D to increase the dimensionality of the data. Consider using a sliding window approach and inserting the sum, product, or a non-linearity of adjacent matrices between pixels. This might be more effective when applied to 2D data (where adjacency is more relevant) before flattening.

<!-- ## License

Coming soon... -->

## Contributing

Coming soon...

