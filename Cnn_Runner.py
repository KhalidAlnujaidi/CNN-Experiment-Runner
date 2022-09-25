# -*- coding: utf-8 -*-
"""The_CNN_Experiment_Runner_finetune.ipynb

# Setup Experiments [ Models, Epochs ]
"""

import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



from datetime import datetime

print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# dictionary of headless model of different architectures to be used for feature extraction

# all from tfhub.dev 

EPOCHS = 100

BATCH_SIZE_LIST = ["32",] # "32", "48", ] #"64", "128"]

models = {
    
    #"efficientnet": ("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1", (224,224)),
    "efficientnet_v2": ("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2", (224,224)),
                     
    #"mobilenet_v2": ("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", (224,224)),
    #"mobilenet_v3": ("https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5", (224,224)),

#    "inception_v1":("https://tfhub.dev/google/imagenet/inception_v1/feature_vector/5", (224,224)),
 #   "inception_v2":("https://tfhub.dev/google/imagenet/inception_v2/feature_vector/5", (224,224)),

    #"inception_v3_inaturalist":("https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5", (299,299)), # trained on iNaturalist dataset
  #  "inception_v3_imagenet": ("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5", (299,299)),

   #"inception_resnet_v2": ("https://tfhub.dev/google/imagenet/inception_resnet_v2/feature_vector/5", (299,299)),


    #"resnet_v2": ("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5", (224,224))

}

for model in models:

  print(f"{model} url --> {models.get(model)[0]} || image size --> {models.get(model)[1]}")

"""## Loading The Data"""

# Commented out IPython magic to ensure Python compatibility.
# Mount Google Drive and get data
# from google.colab import drive
# import pathlib

# drive.mount('/content/drive')
# %cd /content/drive/MyDrive/

# ^--- paint to where images are

# Declare some values

train_path = './image_data/train'
valid_path = './image_data/val'
test_path = './image_data/test'

# ^-- point to proper directory 

IMAGE_SIZE = () # <-- variable initionlization only

# Helper function to make plots and save


def plot_acc(model,dir):
  
  plt.plot(hist['accuracy'])
  plt.plot(hist['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
  plt.savefig(dir)
  plt.clf()
  
def plot_loss(model,dir):
  
  plt.plot(hist['loss'])
  plt.plot(hist['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['Training loss', 'Validation loss'], loc='upper right')
  plt.savefig(dir)
  plt.clf()

"""# Training"""

start_time = datetime.now() # used to print time taken to run all experiments at end.

# loop through all the models

for model in models:
# initonlizing model and batch size
  model_name = model
  feature_extractor = models.get(model)[0]
  IMAGE_SIZE = models.get(model)[1]
  

  # loop through all the batch_sizes for each of the models
  for BATCH_SIZE in BATCH_SIZE_LIST:
    
    BATCH_SIZE = int(BATCH_SIZE)

    # load the data
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        label_mode="categorical",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=1)
    
    class_names = tuple(train_ds.class_names)
    train_size = train_ds.cardinality().numpy()
    train_ds = train_ds.unbatch().batch(int(BATCH_SIZE))
    train_ds = train_ds.repeat()

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_path,
        label_mode="categorical",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=1)
    valid_size = val_ds.cardinality().numpy()
    val_ds = val_ds.unbatch().batch(int(BATCH_SIZE)) 

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        label_mode="categorical",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=1)
    

    # normalization and data augmentation
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    preprocessing_model = tf.keras.Sequential([normalization_layer])
     
    preprocessing_model.add(
          tf.keras.layers.RandomRotation(40))
    preprocessing_model.add(
        tf.keras.layers.RandomTranslation(0, 0.2))
    preprocessing_model.add(
        tf.keras.layers.RandomTranslation(0.2, 0))
    preprocessing_model.add(
        tf.keras.layers.RandomZoom(0.2, 0.2))
    preprocessing_model.add(
        tf.keras.layers.RandomFlip(mode="horizontal"))
    preprocessing_model.add(
        tf.keras.layers.RandomFlip(mode="vertical"))

    train_ds = train_ds.map(lambda images, labels:
                        (preprocessing_model(images), labels))
    val_ds = val_ds.map(lambda images, labels:
                      (normalization_layer(images), labels))
    test_ds = test_ds.map(lambda images, labels:
                      (normalization_layer(images), labels))


    # build the model
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(feature_extractor, trainable=True), # Here model is being downloaded
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(len(class_names),
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    ])
    model.build((None,)+IMAGE_SIZE+(3,))
    model.summary() 
    
    # compile model
    model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9), 
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
    metrics=['accuracy'])


    # Saving best model settings
    SaveModelDir= f"experiments_finetune/{model_name}/exp_for_{str(BATCH_SIZE)}_BatchSize"
    checkpoint_path = SaveModelDir+"/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    metric = 'accuracy'
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor=metric, 
                                                   save_weights_only=False, save_best_only = True, verbose=1)
    # Save complete Model
    model.save(SaveModelDir+f"/model_for_{str(BATCH_SIZE)}_BatchSize")


    ############################################################################
    # start timer for training 
    start_time = datetime.now()

    steps_per_epoch = train_size // int(BATCH_SIZE)
    validation_steps = valid_size // int(BATCH_SIZE)

    hist = model.fit(
      train_ds,
      epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
      validation_data=val_ds,
      validation_steps=validation_steps,
      callbacks = [cp_callback]
      ).history

    end_time = datetime.now() # end timer

    f = open(SaveModelDir+f"/time_to_train_for_{BATCH_SIZE}_model.txt", "x")
    f.write('Duration: {}'.format(end_time - start_time))
    f.close() 
    ############################################################################

    # plot acc
    graph_path = SaveModelDir+'/model_acc.png'
    plot_acc(hist,graph_path)

    # plot loss
    graph_path = SaveModelDir+'/model_loss.png'
    plot_loss(hist,graph_path)




    # Get metracis for val_ds 
    y_pred = []  
    y_true = []

    for image_batch, label_batch in val_ds:   
      y_true.append(label_batch)
      preds = model.predict(image_batch)
      y_pred.append(np.argmax(preds, axis = - 1))
    
    correct_labels = tf.concat([item for item in y_true], axis = 0)
    predicted_labels = tf.concat([item for item in y_pred], axis = 0)
    # Un-One-hot encode the correct labels
    correct_labels_tonums = np.argmax(correct_labels,axis=1)

    cm = confusion_matrix(correct_labels_tonums, predicted_labels)
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_subplot(1,1,1)
    sns.set(font_scale=1.4) #for label size
    sns.heatmap(cm, annot=True, annot_kws={"size": 12},
      cbar = False, cmap='Purples');
    ax1.set_ylabel('True Values',fontsize=14)
    ax1.set_xlabel('Predicted Values',fontsize=14)
    plt.savefig(SaveModelDir+'/con_matrx_val_ds.png')
    plt.clf()

    loss, acc = model.evaluate(val_ds, verbose=2)
    f = open(SaveModelDir+f"/VAL_DS_{BATCH_SIZE}_model.txt", "x")
    f.write(classification_report(correct_labels_tonums,predicted_labels))
    f.write("\n model accuracy on val_ds {:5.2f}%".format(100 * acc))
    f.close() 



    # Get metracis for test_ds
    y_pred = []  
    y_true = [] 

    for image_batch, label_batch in test_ds:   
      y_true.append(label_batch)
      preds = model.predict(image_batch)
      y_pred.append(np.argmax(preds, axis = - 1))

    # convert the true and predicted labels into tensors
    correct_labels = tf.concat([item for item in y_true], axis = 0)
    predicted_labels = tf.concat([item for item in y_pred], axis = 0)

    # Un-One-hot encode the correct labels
    correct_labels_tonums = np.argmax(correct_labels,axis=1)

    
    cm = confusion_matrix(correct_labels_tonums, predicted_labels)
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_subplot(1,1,1)
    sns.set(font_scale=1.4) #for label size
    sns.heatmap(cm, annot=True, annot_kws={"size": 12},
      cbar = False, cmap='Purples');
    ax1.set_ylabel('True Values',fontsize=14)
    ax1.set_xlabel('Predicted Values',fontsize=14)
    plt.savefig(SaveModelDir+'/con_matrx_test_ds.png')
    plt.clf()


    loss, acc = model.evaluate(test_ds, verbose=2)
    f = open(SaveModelDir+f"/TEST_DS_{BATCH_SIZE}_model.txt", "x")
    f.write(classification_report(correct_labels_tonums,predicted_labels))
    f.write("\n model accuracy on test_ds {:5.2f}%".format(100 * acc))
    f.close() 


end_time = datetime.now() # end timer

print("training 3 epochs of this script took : " + 'Duration: {}'.format(end_time - start_time))