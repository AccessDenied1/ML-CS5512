import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

csv_data  = pd.read_csv("train.csv")
train_data = np.array(csv_data)
print("no of training data = ",len(train_data))
W = [0]*11
Xx = [row[0] for row in train_data]
Y = [row[1] for row in train_data]
plt.title("The input data")
plt.scatter(Xx,Y)
plt.show()
#feature matrix
x_fet = np.zeros((len(Xx),11))
for j in range(len(Xx)):
    x = train_data[j][0]
    x_fet[j] = np.array([x**i for i in range(11)])
X = x_fet.copy()
#data spliting
X_tr, X_te, y_tr, y_te = train_test_split(X, Y, test_size=0.20, random_state=42)
Xtr_t = X_tr.transpose()
def ridge(lam):
    I=np.ones((11,11))
    XtX = np.dot(Xtr_t,X_tr)+(lam*I)
    inv=np.linalg.inv(XtX)
    xy=np.dot(inv,Xtr_t)
    W=np.dot(xy,y_tr)
    l_tr=(np.dot(X_tr,W)-y_tr)**2
    l_te=(np.dot(X_te,W)-y_te)**2
    loss_tr = np.average(l_tr)
    loss_te = np.average(l_te)
    return loss_tr,loss_te
step = [x * 0.1 for x in range(0, 100)]
error_tr = []
error_te = []
min_error_te=1000
min_error_tr=1000
for lam in step:
    tr_err, te_err = ridge(lam)
    error_tr.append(tr_err)
    error_te.append(te_err)
    if(min_error_te>te_err):
        min_error_te = te_err
        b_lam_te = lam
    if(min_error_tr>tr_err):
        min_error_tr = tr_err
        b_lam_tr = lam
plt.title("Training (Lambda vs Mean loss)")
plt.plot(step,error_tr)
plt.show()
plt.title("Validation (Lambda vs Mean loss)")
plt.plot(step,error_te)
plt.show()
print("The best lambda as per Validation = ",b_lam_te)
print("The best lambda as per train = ",b_lam_tr)
l_te=ridge(b_lam_te)
l_tr=ridge(b_lam_tr)
print("So the final average loss for lambda = ",b_lam_te ," is",l_te)
print("So the final average loss for lambda = ",b_lam_tr, " is",l_tr)
print("The regularization term aims to penalize the weights for going too high (overfitting) and going too low (underfitting).\nWhen you decide to make your lambda negative, the penalty term is indeed turned into a utility term that helps to increase the weights (i.e., overfitting).\nSo I started the lambda from zero")

print("Now choosing the lambda = ",b_lam_tr,"and training on the whole dataset.")
I=np.ones((11,11))
Xt = X.transpose()
XtX = np.dot(Xt,X)+(b_lam_te*I)
inv=np.linalg.inv(XtX)
xy=np.dot(inv,Xt)
W=np.dot(xy,Y)
print("So the final W = ",W)
y_pred = np.dot(X,W)
los = np.average((y_pred-Y)**2)
print("And the mean squared loss = ",los)
plt.title("Fitting of the curve")
plt.scatter(Xx,Y)
plt.plot(Xx,y_pred)
plt.show()


##on test
csv_data  = pd.read_csv("testX.csv")
test_data = np.array(csv_data)
print("no of test_data ",test_data.shape)
x_test_fet= np.zeros((test_data.shape[0],11))
for j in range(test_data.shape[0]):
    x = test_data[j][0]
    x_test_fet[j] = np.array([x**i for i in range(11)])
X_test = x_test_fet.copy()
pred_Y = np.dot(X_test,W)
Xx_test = [row[0] for row in test_data]
plt.title("Test Data Output")
plt.scatter(Xx_test,pred_Y)
plt.show()

#submission

dict = {'Xts': Xx_test, 'Yp': pred_Y} 
	
df = pd.DataFrame(dict) 

# saving the dataframe 
df.to_csv('submission.csv',index=False)

