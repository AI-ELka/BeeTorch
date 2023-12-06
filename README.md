Hello everyone

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
implement things like convolution in D to increase the dimension (we can use smthg like a sliding window and insert between pixels the sum/product/non-linearity of adjacent matrices (better do it on 2D where adjacency is more relevent then flatten ))
