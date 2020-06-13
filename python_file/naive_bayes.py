from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from scipy import ndimage
from sklearn import metrics
from sklearn import neighbors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import gzip
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import time
import warnings
warnings.filterwarnings("ignore")

def read_image_file(filename, images):
    width = 28
    height = 28
    N = images

    f = gzip.open(filename, 'r')
    f.read(16) # skip preamble, 16 bytes
    buffer = f.read(width * height * N) # read in "N" images as binary data
    data = np.frombuffer(buffer, dtype='uint8') # convert binary data to integers : 0 - 255
    data = data.reshape(N, width, height, 1) # reshape to Nx28x28x1 (only 1 color channel, b/w)
    f.close()
    return data

def read_label_file(filename, labels):
    N = labels
    
    f = gzip.open(filename, 'r')
    f.read(8) # skip preamble, 8 bytes
    buffer = f.read(N) # read in "N" labels as binary data
    data = np.frombuffer(buffer, dtype='uint8') # convert binary data to integers : 0 - 255
    f.close()
    
    return data

X_train = read_image_file('train-images-idx3-ubyte.gz', 60_000)
y_train = read_label_file('train-labels-idx1-ubyte.gz', 60_000)
X_test = read_image_file('t10k-images-idx3-ubyte.gz', 10_000)
y_test = read_label_file('t10k-labels-idx1-ubyte.gz', 10_000)

X_train = X_train.reshape(60_000, 28*28)
X_test = X_test.reshape(10_000, 28*28)


digit_cnt_tr = np.zeros(10)
for item in y_train:
    digit_cnt_tr[item] += 1

digit_cnt_te = np.zeros(10)
for item in y_test:
    digit_cnt_te[item] += 1

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,4))

ax1.bar(np.arange(len(digit_cnt_tr)), digit_cnt_tr, color='green')
ax1.set_xticks(np.arange(len(digit_cnt_tr)))
ax1.set_title('Distribution of digits into MNIST Training set')

ax2.bar(np.arange(len(digit_cnt_te)), digit_cnt_te, color='green')
ax2.set_xticks(np.arange(len(digit_cnt_te)))
ax2.set_title('Distribution of digits into MNIST Test set')

plt.show()

print('\n\nMEAN GRAPH')
X = X_train
y = y_train
u = np.zeros((10, 784))
plt.figure(figsize=(20,10))
for i in range(10):
    u[i] = np.mean(X[y==i], 0)
    plt.subplot(2,5,i+1)
    plt.plot(u[i].reshape(28, 28))
    plt.axis('off')
    plt.title(str(i), fontsize=30)
plt.show()


gnb=GaussianNB(priors=None, var_smoothing=1e-09)
gnb.fit(X_train, y_train)

t0 = time.time()
gnb_tre=gnb.predict(X_train)
train_accuracy=gnb.score(X_train,y_train)
print('Train time elapsed: %.2fs' % (time.time()-t0))

t1 = time.time()
gnb_pre=gnb.predict(X_test)
test_accuracy=gnb.score(X_test,y_test)
print('Test time elapsed: %.2fs' % (time.time()-t1))

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))

cm = metrics.confusion_matrix(y_test, gnb_pre)
plt.figure(figsize=(9, 6))
sns.heatmap(cm, linewidths=.9, annot=True, fmt='g')
plt.suptitle('MNIST Confusion Matrix (GaussianNB based on Testing Accuracy)')

print('                  === Classification Report ===')
print(metrics.classification_report(y_test, gnb_pre))

acc = metrics.accuracy_score(y_test,gnb_pre)
print('\nAccuracy of Classifier on Test Images: ',acc)

unique, counts = np.unique(y_test, return_counts=True)
counts=counts.tolist()

a=cm.diagonal()
acc={}
# The accuracy is found using unique digits and their respective count from test set and comparing that with confusion matrix
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
for i in range(0,len(a)):
    e=[]
    e.append((a[i]/counts[i])*100)
    acc[i]=e
column=[]
for i in range(0,10):
    column.append('Digit %d'%i)    
acc=pd.DataFrame(acc)
acc.columns=column
acc=acc.T
acc.columns=['Accuracy']
acc

print('Cross Validation of GaussianNB')
plt.figure(figsize=(16,8))
plt.title('CV_Size vs Accuracy')
plt.xlabel('CV_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(cross_val_score(gnb,X_test,y_test,scoring="accuracy",cv=20))
plt.legend(['Test'])
plt.show()
cross_val_score(gnb,X_test,y_test,scoring="accuracy",cv=20)


# Gaussian Naive-Bayes with no calibration
a = list()
clf = GaussianNB()
clf.fit(X_test, y_test)  # GaussianNB itself does not support sample-weights
print(clf.score(X_test, y_test))
a.append(clf.score(X_test, y_test))

# Gaussian Naive-Bayes with isotonic calibration
clf_isotonic = CalibratedClassifierCV(clf, cv=3, method='isotonic')
clf_isotonic.fit(X_train, y_train)
clf_isotonic.score(X_test, y_test)
print(clf_isotonic.score(X_train, y_train))
a.append(clf_isotonic.score(X_train, y_train))

# Gaussian Naive-Bayes with sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=3, method='sigmoid')
clf_sigmoid.fit(X_test, y_test,)
print(clf_sigmoid.score(X_test,y_test))
a.append(clf_sigmoid.score(X_test,y_test))

b = ['GNB with no calibration', 'GNB with isotonic calibration', 'GNB with sigmoid calibration']
plt.figure(figsize=(16,6))
plt.grid(True)
plt.bar(b,a)
plt.title('Comparision of Gaussian Naive-Bayes with Calibration')
plt.ylabel('Accuracy')
#plt.xticks()
plt.show()


Tn_S = 60_000
Tt_S = 10_000 

test_accuracy = []
train_size = []
time_interval = []
depth_range = 10
t0 = time.time()
for i in range(1, depth_range):
    ti = time.time()
    train_size.append(Tn_S)
    print('Train Size: ' + str(Tn_S) + ' ' + 'Test_Size: ' + str(Tt_S))
    X_train = read_image_file('train-images-idx3-ubyte.gz', Tn_S)
    y_train = read_label_file('train-labels-idx1-ubyte.gz', Tn_S)
    X_test = read_image_file('t10k-images-idx3-ubyte.gz', Tt_S)
    y_test = read_label_file('t10k-labels-idx1-ubyte.gz', Tt_S)
    
    clf_dt_depth = GaussianNB(priors=None, var_smoothing=1e-09)
    
    clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train)
    
    test_pred_dt = clf_dt_depth.predict(X_test.reshape(-1,28*28))

    test_accuracy.append(metrics.accuracy_score(y_test, test_pred_dt))
    print(test_accuracy)
    print('Train Size: {}/{} and took %.2fs'.format(i,depth_range) % (time.time()-ti))
    Tn_S = Tn_S/2 
    Tn_S = int(Tn_S)
    time_interval.append(time.time()-ti)
print(clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train))
print('Time elapsed: %.2fs' % (time.time()-t0))

plt.figure(figsize=(16,8))
plt.title('Train_Size vs Accuracy')
plt.xlabel('Train_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(train_size,test_accuracy, lw=3)
plt.legend(['Test'])
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Train_Size vs Accuracy 3D')
ax.set_xlabel('Train_Size')
ax.set_ylabel('Test_Accuracy')
ax.set_zlabel('Time_Interval')
ax.plot3D(train_size,test_accuracy,time_interval)


Tn_S = 60_000
Tt_S = 10_000 

test_accuracy = []
test_size = []
time_interval = []
depth_range = 6
t0 = time.time()
for i in range(1, depth_range):
    ti = time.time()
    test_size.append(Tt_S)
    print('Train Size: ' + str(Tn_S) + ' ' + 'Test_Size: ' + str(Tt_S))
    X_train = read_image_file('train-images-idx3-ubyte.gz', Tn_S)
    y_train = read_label_file('train-labels-idx1-ubyte.gz', Tn_S)
    X_test = read_image_file('t10k-images-idx3-ubyte.gz', Tt_S)
    y_test = read_label_file('t10k-labels-idx1-ubyte.gz', Tt_S)
    
    clf_dt_depth = GaussianNB(priors=None, var_smoothing=1e-09)
    
    clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train)
    
    test_pred_dt = clf_dt_depth.predict(X_test.reshape(-1,28*28))

    test_accuracy.append(metrics.accuracy_score(y_test, test_pred_dt))
    print(test_accuracy)
    print('Train Size: {}/{} and took %.2fs'.format(i,depth_range) % (time.time()-ti))
    Tt_S = Tt_S/2 
    Tt_S = int(Tt_S)
    time_interval.append(time.time()-ti)
print(clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train))
print('Time elapsed: %.2fs' % (time.time()-t0))

plt.figure(figsize=(16,8))
plt.title('Test_Size vs Accuracy')
plt.xlabel('Test_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(test_size,test_accuracy, lw=3)
plt.legend(['Test'])
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Test_Size vs Accuracy 3D')
ax.set_xlabel('Test_Size')
ax.set_ylabel('Test_Accuracy')
ax.set_zlabel('Time_Interval')
ax.plot3D(test_size,test_accuracy,time_interval)


# MNB 

X_train = read_image_file('train-images-idx3-ubyte.gz', 60_000)
y_train = read_label_file('train-labels-idx1-ubyte.gz', 60_000)
X_test = read_image_file('t10k-images-idx3-ubyte.gz', 10_000)
y_test = read_label_file('t10k-labels-idx1-ubyte.gz', 10_000)

X_train = X_train.reshape(60_000, 28*28)
X_test = X_test.reshape(10_000, 28*28)

mnb=MultinomialNB()
mnb.fit(X_train, y_train)

t0 = time.time()
mnb_tre=mnb.predict(X_train)
train_accuracy=mnb.score(X_train,y_train)
print('Train time elapsed: %.2fs' % (time.time()-t0))

t1 = time.time()
mnb_pre=mnb.predict(X_test)
test_accuracy=mnb.score(X_test,y_test)
print('Test time elapsed: %.2fs' % (time.time()-t1))

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))


cm = metrics.confusion_matrix(y_test, mnb_pre)
plt.figure(figsize=(9, 6))
sns.heatmap(cm, linewidths=.9, annot=True, fmt='g')
plt.suptitle('MNIST Confusion Matrix (MultinomialNB based on Testing Accuracy)')

print('                  === Classification Report ===')
print(metrics.classification_report(y_test, mnb_pre))

acc = metrics.accuracy_score(y_test,mnb_pre)
print('\nAccuracy of Classifier on Test Images: ',acc)

unique, counts = np.unique(y_test, return_counts=True)
counts=counts.tolist()

a=cm.diagonal()
acc={}
# The accuracy is found using unique digits and their respective count from test set and comparing that with confusion matrix
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
for i in range(0,len(a)):
    e=[]
    e.append((a[i]/counts[i])*100)
    acc[i]=e
column=[]
for i in range(0,10):
    column.append('Digit %d'%i)    
acc=pd.DataFrame(acc)
acc.columns=column
acc=acc.T
acc.columns=['Accuracy']
acc

print('Cross Validation of MultinomialNB')
plt.figure(figsize=(16,8))
plt.title('CV_Size vs Accuracy')
plt.xlabel('CV_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(cross_val_score(mnb,X_test,y_test,scoring="accuracy",cv=20))
plt.legend(['Test'])
plt.show()
print('Max Accuracy: ' + str(max(cross_val_score(mnb,X_test,y_test,scoring="accuracy",cv=20))))
cross_val_score(mnb,X_test,y_test,scoring="accuracy",cv=20)


# Multinomial Naive-Bayes with no calibration
a = list()
clf = MultinomialNB()
clf.fit(X_test, y_test)  # MultinomialNB itself does not support sample-weights
print(clf.score(X_test, y_test))
a.append(clf.score(X_test, y_test))

# Multinomial Naive-Bayes with isotonic calibration
clf_isotonic = CalibratedClassifierCV(clf, cv=3, method='isotonic')
clf_isotonic.fit(X_train, y_train)
clf_isotonic.score(X_test, y_test)
print(clf_isotonic.score(X_train, y_train))
a.append(clf_isotonic.score(X_train, y_train))

# Multinomial Naive-Bayes with sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=3, method='sigmoid')
clf_sigmoid.fit(X_test, y_test,)
print(clf_sigmoid.score(X_test,y_test))
a.append(clf_sigmoid.score(X_test,y_test))

b = ['MNB with no calibration', 'MNB with isotonic calibration', 'MNB with sigmoid calibration']
plt.figure(figsize=(16,6))
plt.grid(True)
plt.bar(b,a)
plt.title('Comparision of Multinomial Naive-Bayes with Calibration')
plt.ylabel('Accuracy')
#plt.xticks()
plt.show()


Tn_S = 60_000
Tt_S = 10_000 

test_accuracy = []
train_size = []
time_interval = []
depth_range = 10
t0 = time.time()
for i in range(1, depth_range):
    ti = time.time()
    train_size.append(Tn_S)
    print('Train Size: ' + str(Tn_S) + ' ' + 'Test_Size: ' + str(Tt_S))
    X_train = read_image_file('train-images-idx3-ubyte.gz', Tn_S)
    y_train = read_label_file('train-labels-idx1-ubyte.gz', Tn_S)
    X_test = read_image_file('t10k-images-idx3-ubyte.gz', Tt_S)
    y_test = read_label_file('t10k-labels-idx1-ubyte.gz', Tt_S)
    
    clf_dt_depth = MultinomialNB()
    
    clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train)
    
    test_pred_dt = clf_dt_depth.predict(X_test.reshape(-1,28*28))

    test_accuracy.append(metrics.accuracy_score(y_test, test_pred_dt))
    print(test_accuracy)
    print('Train Size: {}/{} and took %.2fs'.format(i,depth_range) % (time.time()-ti))
    Tn_S = Tn_S/2 
    Tn_S = int(Tn_S)
    time_interval.append(time.time()-ti)
print(clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train))
print('Time elapsed: %.2fs' % (time.time()-t0))

plt.figure(figsize=(16,8))
plt.title('Train_Size vs Accuracy')
plt.xlabel('Train_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(train_size,test_accuracy, lw=3)
plt.legend(['Test'])
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Train_Size vs Accuracy 3D')
ax.set_xlabel('Train_Size')
ax.set_ylabel('Test_Accuracy')
ax.set_zlabel('Time_Interval')
ax.plot3D(train_size,test_accuracy,time_interval)


Tn_S = 60_000
Tt_S = 10_000 

test_accuracy = []
test_size = []
time_interval = []
depth_range = 6
t0 = time.time()
for i in range(1, depth_range):
    ti = time.time()
    test_size.append(Tt_S)
    print('Train Size: ' + str(Tn_S) + ' ' + 'Test_Size: ' + str(Tt_S))
    X_train = read_image_file('train-images-idx3-ubyte.gz', Tn_S)
    y_train = read_label_file('train-labels-idx1-ubyte.gz', Tn_S)
    X_test = read_image_file('t10k-images-idx3-ubyte.gz', Tt_S)
    y_test = read_label_file('t10k-labels-idx1-ubyte.gz', Tt_S)
    
    clf_dt_depth = MultinomialNB()
    
    clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train)
    
    test_pred_dt = clf_dt_depth.predict(X_test.reshape(-1,28*28))

    test_accuracy.append(metrics.accuracy_score(y_test, test_pred_dt))
    print(test_accuracy)
    print('Train Size: {}/{} and took %.2fs'.format(i,depth_range) % (time.time()-ti))
    Tt_S = Tt_S/2 
    Tt_S = int(Tt_S)
    time_interval.append(time.time()-ti)
print(clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train))
print('Time elapsed: %.2fs' % (time.time()-t0))

plt.figure(figsize=(16,8))
plt.title('Test_Size vs Accuracy')
plt.xlabel('Test_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(test_size,test_accuracy, lw=3)
plt.legend(['Test'])
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Test_Size vs Accuracy 3D')
ax.set_xlabel('Test_Size')
ax.set_ylabel('Test_Accuracy')
ax.set_zlabel('Time_Interval')
ax.plot3D(test_size,test_accuracy,time_interval)

# CNB


X_train = read_image_file('train-images-idx3-ubyte.gz', 60_000)
y_train = read_label_file('train-labels-idx1-ubyte.gz', 60_000)
X_test = read_image_file('t10k-images-idx3-ubyte.gz', 10_000)
y_test = read_label_file('t10k-labels-idx1-ubyte.gz', 10_000)

X_train = X_train.reshape(60_000, 28*28)
X_test = X_test.reshape(10_000, 28*28)

cnb=ComplementNB()
cnb.fit(X_train, y_train)

t0 = time.time()
cnb_tre=cnb.predict(X_train)
train_accuracy=cnb.score(X_train,y_train)
print('Train time elapsed: %.2fs' % (time.time()-t0))

t1 = time.time()
cnb_pre=cnb.predict(X_test)
test_accuracy=cnb.score(X_test,y_test)
print('Test time elapsed: %.2fs' % (time.time()-t1))

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))


cm = metrics.confusion_matrix(y_test, cnb_pre)
plt.figure(figsize=(9, 6))
sns.heatmap(cm, linewidths=.9, annot=True, fmt='g')
plt.suptitle('MNIST Confusion Matrix (ComplementNB based on Testing Accuracy)')

print('                  === Classification Report ===')
print(metrics.classification_report(y_test, cnb_pre))

acc = metrics.accuracy_score(y_test,cnb_pre)
print('\nAccuracy of Classifier on Test Images: ',acc)

unique, counts = np.unique(y_test, return_counts=True)
counts=counts.tolist()

a=cm.diagonal()
acc={}
# The accuracy is found using unique digits and their respective count from test set and comparing that with confusion matrix
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
for i in range(0,len(a)):
    e=[]
    e.append((a[i]/counts[i])*100)
    acc[i]=e
column=[]
for i in range(0,10):
    column.append('Digit %d'%i)    
acc=pd.DataFrame(acc)
acc.columns=column
acc=acc.T
acc.columns=['Accuracy']
acc

print('Cross Validation of ComplementNB')
plt.figure(figsize=(16,8))
plt.title('CV_Size vs Accuracy')
plt.xlabel('CV_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(cross_val_score(cnb,X_test,y_test,scoring="accuracy",cv=20))
plt.legend(['Test'])
plt.show()
print('Max Accuracy: ' + str(max(cross_val_score(cnb,X_test,y_test,scoring="accuracy",cv=20))))
cross_val_score(cnb,X_test,y_test,scoring="accuracy",cv=20)


# Complement Naive-Bayes with no calibration
a = list()
clf = ComplementNB()
clf.fit(X_test, y_test)  # MultinomialNB itself does not support sample-weights
print(clf.score(X_test, y_test))
a.append(clf.score(X_test, y_test))

# Complement Naive-Bayes with isotonic calibration
clf_isotonic = CalibratedClassifierCV(clf, cv=3, method='isotonic')
clf_isotonic.fit(X_train, y_train)
clf_isotonic.score(X_test, y_test)
print(clf_isotonic.score(X_train, y_train))
a.append(clf_isotonic.score(X_train, y_train))

# Complement Naive-Bayes with sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=3, method='sigmoid')
clf_sigmoid.fit(X_test, y_test,)
print(clf_sigmoid.score(X_test,y_test))
a.append(clf_sigmoid.score(X_test,y_test))

b = ['CNB with no calibration', 'CNB with isotonic calibration', 'CNB with sigmoid calibration']
plt.figure(figsize=(16,6))
plt.grid(True)
plt.bar(b,a)
plt.title('Comparision of Multinomial Naive-Bayes with Calibration')
plt.ylabel('Accuracy')
#plt.xticks()
plt.show()


Tn_S = 60_000
Tt_S = 10_000 

test_accuracy = []
train_size = []
time_interval = []
depth_range = 10
t0 = time.time()
for i in range(1, depth_range):
    ti = time.time()
    train_size.append(Tn_S)
    print('Train Size: ' + str(Tn_S) + ' ' + 'Test_Size: ' + str(Tt_S))
    X_train = read_image_file('train-images-idx3-ubyte.gz', Tn_S)
    y_train = read_label_file('train-labels-idx1-ubyte.gz', Tn_S)
    X_test = read_image_file('t10k-images-idx3-ubyte.gz', Tt_S)
    y_test = read_label_file('t10k-labels-idx1-ubyte.gz', Tt_S)
    
    clf_dt_depth = ComplementNB()
    
    clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train)
    
    test_pred_dt = clf_dt_depth.predict(X_test.reshape(-1,28*28))

    test_accuracy.append(metrics.accuracy_score(y_test, test_pred_dt))
    print(test_accuracy)
    print('Train Size: {}/{} and took %.2fs'.format(i,depth_range) % (time.time()-ti))
    Tn_S = Tn_S/2 
    Tn_S = int(Tn_S)
    time_interval.append(time.time()-ti)
print(clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train))
print('Time elapsed: %.2fs' % (time.time()-t0))

plt.figure(figsize=(16,8))
plt.title('Train_Size vs Accuracy')
plt.xlabel('Train_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(train_size,test_accuracy, lw=3)
plt.legend(['Test'])
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Train_Size vs Accuracy 3D')
ax.set_xlabel('Train_Size')
ax.set_ylabel('Test_Accuracy')
ax.set_zlabel('Time_Interval')
ax.plot3D(train_size,test_accuracy,time_interval)


Tn_S = 60_000
Tt_S = 10_000 

test_accuracy = []
test_size = []
time_interval = []
depth_range = 6
t0 = time.time()
for i in range(1, depth_range):
    ti = time.time()
    test_size.append(Tt_S)
    print('Train Size: ' + str(Tn_S) + ' ' + 'Test_Size: ' + str(Tt_S))
    X_train = read_image_file('train-images-idx3-ubyte.gz', Tn_S)
    y_train = read_label_file('train-labels-idx1-ubyte.gz', Tn_S)
    X_test = read_image_file('t10k-images-idx3-ubyte.gz', Tt_S)
    y_test = read_label_file('t10k-labels-idx1-ubyte.gz', Tt_S)
    
    clf_dt_depth = ComplementNB()
    
    clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train)
    
    test_pred_dt = clf_dt_depth.predict(X_test.reshape(-1,28*28))

    test_accuracy.append(metrics.accuracy_score(y_test, test_pred_dt))
    print(test_accuracy)
    print('Train Size: {}/{} and took %.2fs'.format(i,depth_range) % (time.time()-ti))
    Tt_S = Tt_S/2 
    Tt_S = int(Tt_S)
    time_interval.append(time.time()-ti)
print(clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train))
print('Time elapsed: %.2fs' % (time.time()-t0))

plt.figure(figsize=(16,8))
plt.title('Test_Size vs Accuracy')
plt.xlabel('Test_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(test_size,test_accuracy, lw=3)
plt.legend(['Test'])
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Test_Size vs Accuracy 3D')
ax.set_xlabel('Test_Size')
ax.set_ylabel('Test_Accuracy')
ax.set_zlabel('Time_Interval')
ax.plot3D(test_size,test_accuracy,time_interval)


# # BNB

X_train = read_image_file('train-images-idx3-ubyte.gz', 60_000)
y_train = read_label_file('train-labels-idx1-ubyte.gz', 60_000)
X_test = read_image_file('t10k-images-idx3-ubyte.gz', 10_000)
y_test = read_label_file('t10k-labels-idx1-ubyte.gz', 10_000)

X_train = X_train.reshape(60_000, 28*28)
X_test = X_test.reshape(10_000, 28*28)

bnb=BernoulliNB()
bnb.fit(X_train, y_train)

t0 = time.time()
bnb_tre=bnb.predict(X_train)
train_accuracy=bnb.score(X_train,y_train)
print('Train time elapsed: %.2fs' % (time.time()-t0))

t1 = time.time()
bnb_pre=bnb.predict(X_test)
test_accuracy=bnb.score(X_test,y_test)
print('Test time elapsed: %.2fs' % (time.time()-t1))

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))


cm = metrics.confusion_matrix(y_test, bnb_pre)
plt.figure(figsize=(9, 6))
sns.heatmap(cm, linewidths=.9, annot=True, fmt='g')
plt.suptitle('MNIST Confusion Matrix (BernoulliNB based on Testing Accuracy)')

print('                  === Classification Report ===')
print(metrics.classification_report(y_test, bnb_pre))

acc = metrics.accuracy_score(y_test,bnb_pre)
print('\nAccuracy of Classifier on Test Images: ',acc)

unique, counts = np.unique(y_test, return_counts=True)
counts=counts.tolist()

a=cm.diagonal()
acc={}
# The accuracy is found using unique digits and their respective count from test set and comparing that with confusion matrix
unique_elements, counts_elements = np.unique(y_test, return_counts=True)
for i in range(0,len(a)):
    e=[]
    e.append((a[i]/counts[i])*100)
    acc[i]=e
column=[]
for i in range(0,10):
    column.append('Digit %d'%i)    
acc=pd.DataFrame(acc)
acc.columns=column
acc=acc.T
acc.columns=['Accuracy']
acc


print('Cross Validation of BernoulliNB')
plt.figure(figsize=(16,8))
plt.title('CV_Size vs Accuracy')
plt.xlabel('CV_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(cross_val_score(bnb,X_test,y_test,scoring="accuracy",cv=20))
plt.legend(['Test'])
plt.show()
print('Max Accuracy: ' + str(max(cross_val_score(bnb,X_test,y_test,scoring="accuracy",cv=20))))
cross_val_score(bnb,X_test,y_test,scoring="accuracy",cv=20)

# Complement Naive-Bayes with no calibration
a = list()
clf = BernoulliNB()
clf.fit(X_test, y_test)  # MultinomialNB itself does not support sample-weights
print(clf.score(X_test, y_test))
a.append(clf.score(X_test, y_test))

# Complement Naive-Bayes with isotonic calibration
clf_isotonic = CalibratedClassifierCV(clf, cv=3, method='isotonic')
clf_isotonic.fit(X_train, y_train)
clf_isotonic.score(X_test, y_test)
print(clf_isotonic.score(X_train, y_train))
a.append(clf_isotonic.score(X_train, y_train))

# Complement Naive-Bayes with sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=3, method='sigmoid')
clf_sigmoid.fit(X_test, y_test,)
print(clf_sigmoid.score(X_test,y_test))
a.append(clf_sigmoid.score(X_test,y_test))

b = ['BNB with no calibration', 'BNB with isotonic calibration', 'BNB with sigmoid calibration']
plt.figure(figsize=(16,6))
plt.grid(True)
plt.bar(b,a)
plt.title('Comparision of Bernoulli Naive-Bayes with Calibration')
plt.ylabel('Accuracy')
#plt.xticks()
plt.show()


Tn_S = 60_000
Tt_S = 10_000 

test_accuracy = []
train_size = []
time_interval = []
depth_range = 10
t0 = time.time()
for i in range(1, depth_range):
    ti = time.time()
    train_size.append(Tn_S)
    print('Train Size: ' + str(Tn_S) + ' ' + 'Test_Size: ' + str(Tt_S))
    X_train = read_image_file('train-images-idx3-ubyte.gz', Tn_S)
    y_train = read_label_file('train-labels-idx1-ubyte.gz', Tn_S)
    X_test = read_image_file('t10k-images-idx3-ubyte.gz', Tt_S)
    y_test = read_label_file('t10k-labels-idx1-ubyte.gz', Tt_S)
    
    clf_dt_depth = BernoulliNB()
    
    clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train)
    
    test_pred_dt = clf_dt_depth.predict(X_test.reshape(-1,28*28))

    test_accuracy.append(metrics.accuracy_score(y_test, test_pred_dt))
    print(test_accuracy)
    print('Train Size: {}/{} and took %.2fs'.format(i,depth_range) % (time.time()-ti))
    Tn_S = Tn_S/2 
    Tn_S = int(Tn_S)
    time_interval.append(time.time()-ti)
print(clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train))
print('Time elapsed: %.2fs' % (time.time()-t0))

plt.figure(figsize=(16,8))
plt.title('Train_Size vs Accuracy')
plt.xlabel('Train_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(train_size,test_accuracy, lw=3)
plt.legend(['Test'])
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Train_Size vs Accuracy 3D')
ax.set_xlabel('Train_Size')
ax.set_ylabel('Test_Accuracy')
ax.set_zlabel('Time_Interval')
ax.plot3D(train_size,test_accuracy,time_interval)


Tn_S = 60_000
Tt_S = 10_000 

test_accuracy = []
test_size = []
time_interval = []
depth_range = 6
t0 = time.time()
for i in range(1, depth_range):
    ti = time.time()
    test_size.append(Tt_S)
    print('Train Size: ' + str(Tn_S) + ' ' + 'Test_Size: ' + str(Tt_S))
    X_train = read_image_file('train-images-idx3-ubyte.gz', Tn_S)
    y_train = read_label_file('train-labels-idx1-ubyte.gz', Tn_S)
    X_test = read_image_file('t10k-images-idx3-ubyte.gz', Tt_S)
    y_test = read_label_file('t10k-labels-idx1-ubyte.gz', Tt_S)
    
    clf_dt_depth = BernoulliNB()
    
    clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train)
    
    test_pred_dt = clf_dt_depth.predict(X_test.reshape(-1,28*28))

    test_accuracy.append(metrics.accuracy_score(y_test, test_pred_dt))
    print(test_accuracy)
    print('Train Size: {}/{} and took %.2fs'.format(i,depth_range) % (time.time()-ti))
    Tt_S = Tt_S/2 
    Tt_S = int(Tt_S)
    time_interval.append(time.time()-ti)
print(clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train))
print('Time elapsed: %.2fs' % (time.time()-t0))

plt.figure(figsize=(16,8))
plt.title('Test_Size vs Accuracy')
plt.xlabel('Test_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(test_size,test_accuracy, lw=3)
plt.legend(['Test'])
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Test_Size vs Accuracy 3D')
ax.set_xlabel('Test_Size')
ax.set_ylabel('Test_Accuracy')
ax.set_zlabel('Time_Interval')
ax.plot3D(test_size,test_accuracy,time_interval)


# # CategoricalNB
X_train = read_image_file('train-images-idx3-ubyte.gz', 60_000)
y_train = read_label_file('train-labels-idx1-ubyte.gz', 60_000)
X_test = read_image_file('t10k-images-idx3-ubyte.gz', 10_000)
y_test = read_label_file('t10k-labels-idx1-ubyte.gz', 10_000)

X_train = X_train.reshape(60_000, 28*28)
X_test = X_test.reshape(10_000, 28*28)

ctnb=CategoricalNB()
ctnb.fit(X_train, y_train)

t0 = time.time()
ctnb_tre=ctnb.predict(X_train)
train_accuracy=ctnb.score(X_train,y_train)
print('Train time elapsed: %.2fs' % (time.time()-t0))

t1 = time.time()
#bnb_pre=ctnb.predict(X_test)
#test_accuracy=ctnb.score(X_test,y_test)
#print('Test time elapsed: %.2fs' % (time.time()-t1))

print('Training accuracy: %0.2f%%' % (train_accuracy*100))
#print('Testing accuracy: %0.2f%%' % (test_accuracy*100))


cm = metrics.confusion_matrix(y_train, ctnb_tre)
plt.figure(figsize=(9, 6))
sns.heatmap(cm, linewidths=.9, annot=True, fmt='g')
plt.suptitle('MNIST Confusion Matrix (BernoulliNB based on Testing Accuracy)')

print('                  === Classification Report ===')
print(metrics.classification_report(y_train, ctnb_tre))

acc = metrics.accuracy_score(y_train,ctnb_tre)
print('\nAccuracy of Classifier on Test Images: ',acc)

unique, counts = np.unique(y_train, return_counts=True)
counts=counts.tolist()

a=cm.diagonal()
acc={}
# The accuracy is found using unique digits and their respective count from test set and comparing that with confusion matrix
unique_elements, counts_elements = np.unique(y_train, return_counts=True)
for i in range(0,len(a)):
    e=[]
    e.append((a[i]/counts[i])*100)
    acc[i]=e
column=[]
for i in range(0,10):
    column.append('Digit %d'%i)    
acc=pd.DataFrame(acc)
acc.columns=column
acc=acc.T
acc.columns=['Accuracy']
acc


Tn_S = 60_000
Tt_S = 10_000 

test_accuracy = []
train_size = []
time_interval = []
depth_range = 10
t0 = time.time()
for i in range(1, depth_range):
    ti = time.time()
    train_size.append(Tn_S)
    print('Train Size: ' + str(Tn_S) + ' ' + 'Test_Size: ' + str(Tt_S))
    X_train = read_image_file('train-images-idx3-ubyte.gz', Tn_S)
    y_train = read_label_file('train-labels-idx1-ubyte.gz', Tn_S)
    X_test = read_image_file('t10k-images-idx3-ubyte.gz', Tt_S)
    y_test = read_label_file('t10k-labels-idx1-ubyte.gz', Tt_S)
    
    clf_dt_depth = CategoricalNB()
    
    clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train)
    
    test_pred_dt = clf_dt_depth.predict(X_train.reshape(-1,28*28))

    test_accuracy.append(metrics.accuracy_score(y_train, test_pred_dt))
    print(test_accuracy)
    print('Train Size: {}/{} and took %.2fs'.format(i,depth_range) % (time.time()-ti))
    Tn_S = Tn_S/2 
    Tn_S = int(Tn_S)
    time_interval.append(time.time()-ti)
print(clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train))
print('Time elapsed: %.2fs' % (time.time()-t0))

plt.figure(figsize=(16,8))
plt.title('Train_Size vs Accuracy')
plt.xlabel('Train_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(train_size,test_accuracy, lw=3)
plt.legend(['Test'])
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Train_Size vs Accuracy 3D')
ax.set_xlabel('Train_Size')
ax.set_ylabel('Test_Accuracy')
ax.set_zlabel('Time_Interval')
ax.plot3D(train_size,test_accuracy,time_interval)

Tn_S = 60_000
Tt_S = 10_000 

test_accuracy = []
test_size = []
time_interval = []
depth_range = 6
t0 = time.time()
for i in range(1, depth_range):
    ti = time.time()
    test_size.append(Tt_S)
    print('Train Size: ' + str(Tn_S) + ' ' + 'Test_Size: ' + str(Tt_S))
    X_train = read_image_file('train-images-idx3-ubyte.gz', Tn_S)
    y_train = read_label_file('train-labels-idx1-ubyte.gz', Tn_S)
    X_test = read_image_file('t10k-images-idx3-ubyte.gz', Tt_S)
    y_test = read_label_file('t10k-labels-idx1-ubyte.gz', Tt_S)
    
    clf_dt_depth = CategoricalNB()
    
    clf_dt_depth.fit(X_test.reshape(-1,28*28), y_test)
    
    test_pred_dt = clf_dt_depth.predict(X_test.reshape(-1,28*28))

    test_accuracy.append(metrics.accuracy_score(y_test, test_pred_dt))
    print(test_accuracy)
    print('Train Size: {}/{} and took %.2fs'.format(i,depth_range) % (time.time()-ti))
    Tt_S = Tt_S/2 
    Tt_S = int(Tt_S)
    time_interval.append(time.time()-ti)
print(clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train))
print('Time elapsed: %.2fs' % (time.time()-t0))

plt.figure(figsize=(16,8))
plt.title('Test_Size vs Accuracy')
plt.xlabel('Test_Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.plot(test_size,test_accuracy, lw=3)
plt.legend(['Test'])
plt.show()

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Test_Size vs Accuracy 3D')
ax.set_xlabel('Test_Size')
ax.set_ylabel('Test_Accuracy')
ax.set_zlabel('Time_Interval')
ax.plot3D(test_size,test_accuracy,time_interval)



