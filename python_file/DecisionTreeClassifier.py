from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV
from scipy import ndimage
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import gzip
import matplotlib.pyplot as plt
import time
import seaborn as sns
import pandas as pd


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



bnb=DecisionTreeClassifier()
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
plt.suptitle('MNIST Confusion Matrix (DecisionTreeClassifier based on Testing Accuracy)')

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



correct_indices = np.nonzero(bnb_pre == y_test)[0]
incorrect_indices = np.nonzero(bnb_pre != y_test)[0]


num_figures = 6
plt.figure(figsize=(16,8))
for i, correct in enumerate(correct_indices[:num_figures]):
    plt.subplot(1,num_figures,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(bnb_pre[correct], y_test[correct]))



num_figures = 6
plt.figure(figsize=(16,8))
for i, incorrect in enumerate(incorrect_indices[:num_figures]):
    plt.subplot(1,num_figures,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(bnb_pre[incorrect], y_test[incorrect]))




test_accuracy = []
t0 = time.time()
depth_range = 10
for i in range(1, depth_range):
    ti = time.time()
    clf_dt_depth = DecisionTreeClassifier(max_depth=i)
    clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train)
    test_pred_dt = clf_dt_depth.predict(X_test.reshape(-1,28*28))
    print(test_accuracy)
    test_accuracy.append(accuracy_score(y_test, test_pred_dt))
    print('Depth: {}/{} and took %.2fs'.format(i,depth_range) % (time.time()-ti))
print(clf_dt_depth.fit(X_train.reshape(-1,28*28), y_train))
print('Time elapsed: %.2fs' % (time.time()-t0))



plt.figure(figsize=(14,6))
plt.title('Accuracy vs. Depth')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.plot(test_accuracy)
plt.legend(['Test'])
plt.show()



print('Highest test accuracy: {} at depth of {}'
      .format(max(test_accuracy), test_accuracy.index(max(test_accuracy))+1))



clf_dt_best_depth = DecisionTreeClassifier(max_depth=test_accuracy.index(max(test_accuracy))+1)
clf_dt_best_depth.fit(X_train.reshape(-1,28*28), y_train)
pred_dt_best_depth = clf_dt_best_depth.predict(X_test.reshape(-1,28*28))



cnf_matrix_bd = confusion_matrix(y_test,pred_dt_best_depth)
plt.figure(figsize=(5,5))
plt.imshow(cnf_matrix_bd, cmap='plasma')
plt.colorbar()
plt.title("Confusion Matrix at Best Depth of {}".format(test_accuracy.index(max(test_accuracy))+1))


print(classification_report(y_test,pred_dt_best_depth))

correct_indices_d = np.nonzero(pred_dt_best_depth == y_test)[0]
incorrect_indices_d = np.nonzero(pred_dt_best_depth != y_test)[0]

num_figures = 6
plt.figure(figsize=(16,8))
for i, correct in enumerate(correct_indices_d[:num_figures]):
    plt.subplot(1,num_figures,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(pred_dt_best_depth[correct], y_test[correct]))


num_figures = 6
plt.figure(figsize=(16,8))
for i, incorrect in enumerate(incorrect_indices_d[:num_figures]):
    plt.subplot(1,num_figures,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(pred_dt_best_depth[incorrect], y_test[incorrect]))


importances_dt = bnb.feature_importances_
importances_depth = clf_dt_best_depth.feature_importances_

heat_dt = importances_dt.reshape(28,28)
heat_depth = importances_depth.reshape(28,28)
heat_dif = heat_dt-heat_depth
plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
plt.title('Default Decision Tree')
plt.imshow(heat_dt, cmap='plasma', interpolation='nearest')
plt.colorbar()
plt.subplot(1,3,2)
plt.title('Decision Tree at Depth {}'.format(test_accuracy.index(max(test_accuracy))+1))
plt.imshow(heat_depth, cmap='plasma', interpolation='nearest')
plt.colorbar()
plt.subplot(1,3,3)
plt.title('Difference')
plt.imshow(heat_dif, cmap='plasma', interpolation='nearest')
plt.colorbar()
plt.show()


add_in_pixels = False


def add_custom_features(features):
    new_features = []
    for feature in features:
        if add_in_pixels:
            tmp_feature = feature.reshape(28*28)
            tmp_feature = tmp_feature.tolist()
        else:
            tmp_feature = []
        tmp_feature.append(np.mean(feature))  # Find the average pixel value
        tmp_feature.append(np.count_nonzero(feature.tolist()))  # Count non-zero values
        center = ndimage.measurements.center_of_mass(feature.reshape(28,28))
        tmp_feature.append(np.sum(center))  # Sum of center of mass
        tmp_feature.append(np.mean(center))  # Mean of center of mass
        new_features.append(tmp_feature)
    return np.array(new_features)


from scipy import ndimage
X_train_new = add_custom_features(X_train)
X_test_new = add_custom_features(X_test)


t0 = time.time()
clf_dt_new = DecisionTreeClassifier()
clf_dt_new.fit(X_train_new, y_train)
print('Time elapsed: %.2fs' % (time.time()-t0))

pred_dt_new = clf_dt_new.predict(X_test_new)
print('Predicted', len(bnb_pre), 'digits with accuracy:', accuracy_score(y_test, pred_dt_new))



if add_in_pixels:
    importances_dt_new = clf_dt_new.feature_importances_[784:]
else:
    importances_dt_new = clf_dt_new.feature_importances_
print('Importanes: {}'.format(importances_dt_new))



plt.figure(figsize=(16,6))
ind = np.arange(len(importances_dt_new))
plt.bar(ind, importances_dt_new, 0.35)
plt.title('Comparision of Custom Feature Importance')
plt.ylabel('Importance')
plt.xticks(ind, ('Average Pixel Value', '# Non-Zero Pixels', 'Sum of Centroid', 'Mean of Centroid'))
plt.show()


cnf_matrix_custom = confusion_matrix(y_test,pred_dt_new)
plt.figure(figsize=(5,5))
plt.imshow(cnf_matrix_custom, cmap='plasma')
plt.colorbar()
plt.title("Confusion Matrix Custom Features")
plt.show()


print(classification_report(y_test,pred_dt_new))

