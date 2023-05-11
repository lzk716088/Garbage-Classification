import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import urllib
import itertools
import random, os, glob
from imutils import paths
from sklearn.utils import shuffle
from urllib.request import urlopen
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import  ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import glob, os, random

target_size = (300, 300)
model = tf.keras.models.load_model('mymodel.h5')

def model_testing(model,path):
  img = image.load_img(path, target_size=(target_size))
  img = image.img_to_array(img, dtype=np.uint8)
  img = np.array(img)/255.0
  p = model.predict(img.reshape(1,300,300,3))
  predicted_class = np.argmax(p[0])
  return img, p, predicted_class

img, p, predicted_class = model_testing(model,r"C:\Users\aiialab\Desktop\siang\ml_dataset\one\14.jpg") 

waste_labels = {0:"cardboard", 1:"glass", 2:"metal", 3:"paper", 4:"plastic", 5:"trash"}
def plot_model_testing(img, p, predicted_class):
  plt.axis("off")
  plt.imshow(img.squeeze())
  plt.title("Maximum Probabilty: " + str(np.max(p[0], axis=-1)) + "\n" + "Predicted Class: " + str(waste_labels[predicted_class]))
  plt.show()

plot_model_testing(img, p, predicted_class)

base_path = r"C:\Users\aiialab\Desktop\siang\ml_dataset\test"  # load testdata的位置

img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))

#資料處理
def CNN_data_preparation():
  train_datagen = ImageDataGenerator(
      horizontal_flip=True,
      vertical_flip=True,
      validation_split=0.1,
      rescale=1./255,
      shear_range=0.1,
      zoom_range=0.1,
      width_shift_range=0.1,
      height_shift_range=0.1
  )
  test_datagen = ImageDataGenerator(
      rescale=1./255,
      validation_split=0.1
  )
  train_generator = train_datagen.flow_from_directory(
      base_path,
      target_size=(300, 300),
      batch_size=32,
      class_mode='categorical',
      subset='training',   
  )
  validation_generator = test_datagen.flow_from_directory(
      base_path,
      target_size=(300, 300),
      batch_size=200,
      class_mode='categorical',
      subset='validation',   
      shuffle=False
  )
  return train_generator, validation_generator
train_generator, validation_generator = CNN_data_preparation()

#標籤
labels = (train_generator.class_indices)
labels = dict((k,v) for k,v in labels.items())
labels_index = dict((v,k) for k,v in labels.items())
print(labels) #{'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
print(labels_index) #{0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

model = tf.keras.models.load_model('mymodel.h5') #load訓練好的model

def CNN_model_testing(model):
    X_test, y_test_labels = validation_generator.next()
    y_pred_labels = model.predict(X_test)
    y_pred = np.argmax(y_pred_labels, axis=1)
    y_test = np.argmax(y_test_labels, axis=1)
    target_names = list(labels.keys())
    print(classification_report(y_test, y_pred, target_names=target_names))
    plt.figure(figsize=(16, 16))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=0.5)
        if labels_index[np.argmax(y_pred_labels[i])] == labels_index[np.argmax(y_test_labels[i])]:
          plt.title('pred:%s / truth:%s' % (labels_index[np.argmax(y_pred_labels[i])], labels_index[np.argmax(y_test_labels[i])]), color='green')
        else:
          plt.title('pred:%s / truth:%s' % (labels_index[np.argmax(y_pred_labels[i])], labels_index[np.argmax(y_test_labels[i])]))
        plt.imshow(X_test[i])
    plt.show()
    plt.figure(figsize=(16, 16))
    incorrect = False
    k=0
    for i in range(len(y_pred)):
        if k==12:
           break
        if labels_index[np.argmax(y_pred_labels[i])] != labels_index[np.argmax(y_test_labels[i])]:
            incorrect = True
            plt.subplot(4, 4, k+1)
            plt.subplots_adjust(left=None, bottom=0.2, right=None, top=None, wspace=None, hspace=0.5)
            plt.title('pred:%s / truth:%s' % (labels_index[np.argmax(y_pred_labels[i])], labels_index[np.argmax(y_test_labels[i])]))
            plt.imshow(X_test[i])
            k+=1
    if incorrect:
        plt.show()
    else:
        print("All predictions are correct!")
    return y_test, y_pred
y_test, y_pred = CNN_model_testing(model)

cm = confusion_matrix(y_test, y_pred)
def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.OrRd):
  if normalize:
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
  plt.figure(figsize=(8,5))
  plt.imshow(cm, interpolation="nearest", cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)
  fmt = ".2f" if normalize else "d"
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
  plt.tight_layout()
  plt.ylabel("True Labels", fontweight="bold")
  plt.xlabel("Predicted Labels", fontweight="bold")
  plt.show()
plot_confusion_matrix(cm, labels.keys())