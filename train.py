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
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import LSTM

base_path = r"C:\Users\aiialab\Desktop\siang\ml_dataset\train"  # load traindata的位置

img_list = glob.glob(os.path.join(base_path, '*/*.jpg'))

print(len(img_list)) # 總共多少張圖片

# 查看原始圖片
folder_names = os.listdir(base_path) # 取得所有資料夾名稱
for i, folder_name in enumerate(folder_names):
    folder_path = os.path.join(base_path, folder_name)
    img_paths = os.listdir(folder_path)
    img_path = os.path.join(folder_path, random.choice(img_paths))
    img = load_img(img_path)
    img = img_to_array(img, dtype=np.uint8)
    plt.subplot(2, 3, i+1)
    plt.title(folder_name)
    plt.imshow(img.squeeze())
plt.show()

def CNN_data_preparation():
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2
    )
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    train_generator = train_datagen.flow_from_directory(
        base_path,
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        base_path,
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    return train_generator, validation_generator, test_generator

base_path = r"C:\Users\aiialab\Desktop\siang\ml_dataset\train"
test_path = r"C:\Users\aiialab\Desktop\siang\ml_dataset\test"

train_generator, validation_generator, test_generator = CNN_data_preparation()

#標籤
labels = (train_generator.class_indices)
labels = dict((k,v) for k,v in labels.items())
labels_index = dict((v,k) for k,v in labels.items())

print(labels) #{'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}
print(labels_index) #{0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}


def CNN_create_and_fit_model(train_generator, validation_generator, summary=True, fit=True, epochs=150):
  model = Sequential()
  model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(300, 300, 3))),
  model.add(MaxPooling2D(pool_size=2)),

  model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')),
  model.add(MaxPooling2D(pool_size=2)),

  model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')),
  model.add(MaxPooling2D(pool_size=2)),

  model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')),
  model.add(MaxPooling2D(pool_size=2)),

  model.add(Flatten()),

  model.add(Dense(64, activation='relu')),

  model.add(Dense(6, activation='softmax')),

  model.compile(loss="categorical_crossentropy", 
                optimizer="adam", 
                metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "acc"])
  
  # EarlyStopping 是一個回調函數，若連續30個 epoch 都沒有提升，則停止訓練
  # ModelCheckpoint 也是一個回調函數，用於保存訓練過程中最好的模型。在此設置監控的指標是驗證集上的損失函數，每當有更好的模型時，就會將模型保存到指定路徑。
  callbacks = [EarlyStopping(monitor="val_loss", patience=30, verbose=1, mode="min"), 
              ModelCheckpoint(filepath="mymodel_cnn.h5", monitor="val_loss", mode="min", save_best_only=True, save_weights_only=False, verbose=1)]

  if summary:
      model.summary()

  if fit:
    history = model.fit_generator(generator=train_generator, epochs=epochs, validation_data=validation_generator, 
                                callbacks=callbacks, workers=4, steps_per_epoch=10, validation_steps=251//32)
  return history
history = CNN_create_and_fit_model(train_generator, validation_generator)

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
def VGG16_create_and_fit_model(train_generator, validation_generator, summary=True, fit=True, epochs=150):
  base_model = VGG16(include_top=False, weights='imagenet', input_shape=(300, 300, 3))

  # freeze the base model
  for layer in base_model.layers:
    layer.trainable = False

  # add new layers
  x = base_model.output
  x = Flatten()(x)
  x = Dense(64, activation='relu')(x)
  predictions = Dense(6, activation='softmax')(x)

  # create the model
  model = Model(inputs=base_model.input, outputs=predictions)

  # compile the model
  model.compile(loss="categorical_crossentropy", 
                optimizer="adam", 
                metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "acc"])

  # EarlyStopping is a callback function to stop the training if the validation loss does not improve for 30 consecutive epochs
  # ModelCheckpoint is a callback function to save the best model during training based on validation loss
  callbacks = [EarlyStopping(monitor="val_loss", patience=30, verbose=1, mode="min"), 
              ModelCheckpoint(filepath="mymodel_VGG.h5", monitor="val_loss", mode="min", save_best_only=True, save_weights_only=False, verbose=1)]

  if summary:
      model.summary()

  if fit:
    history = model.fit_generator(generator=train_generator, epochs=epochs, validation_data=validation_generator, 
                                callbacks=callbacks, workers=4, steps_per_epoch=10, validation_steps=251//32)
  return history
history = VGG16_create_and_fit_model(train_generator, validation_generator)

from tensorflow.keras.applications import ResNet50V2
def ResNet50V2_create_and_fit_model(train_generator, validation_generator, summary=True, fit=True, epochs=150):
    model = Sequential()
    conv_base = ResNet50V2(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    conv_base.trainable = False

    model.compile(loss="categorical_crossentropy", 
                  optimizer="adam", 
                  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "acc"])
    
    callbacks = [EarlyStopping(monitor="val_loss", patience=30, verbose=1, mode="min"), 
                ModelCheckpoint(filepath="mymodel_ResNet.h5", monitor="val_loss", mode="min", save_best_only=True, save_weights_only=False, verbose=1)]

    if summary:
        model.summary()

    if fit:
        history = model.fit_generator(generator=train_generator, epochs=epochs, validation_data=validation_generator, 
                                    callbacks=callbacks, workers=4, steps_per_epoch=10, validation_steps=251//32)
    return history
history = ResNet50V2_create_and_fit_model(train_generator, validation_generator)

model = tf.keras.models.load_model('mymodel.h5')

#評估模型
def CNN_model_evaluate(model):
    loss, precision, recall, acc = model.evaluate(validation_generator, batch_size=32)
    print("Test Accuracy: %.2f" % (acc))
    print("Test Loss: %.2f" % (loss))
    print("Test Precision: %.2f" % (precision))
    print("Test Recall: %.2f" % (recall))
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history["acc"], color="r", label="Training Accuracy")
    plt.plot(history.history["val_acc"], color="b", label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.ylim([min(plt.ylim()),1])
    plt.title("Training and Validation Accuracy", fontsize=16)
    plt.show()
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], color="r", label="Training Loss")
    plt.plot(history.history["val_loss"], color="b", label="Validation Loss")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.ylim([0, max(plt.ylim())])
    plt.title("Training and Validation Loss", fontsize=16)
    plt.show()
CNN_model_evaluate(model)

def CNN_model_testing(model):
    y_test_labels = test_generator.labels
    y_pred_labels = model.predict(test_generator)
    y_pred = np.argmax(y_pred_labels, axis=1)
    y_test = y_test_labels
    target_names = list(labels.keys())
    print(classification_report(y_test, y_pred, target_names=target_names))
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

