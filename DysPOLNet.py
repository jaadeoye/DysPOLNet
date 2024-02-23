import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import PIL
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from pathlib import Path
import imghdr
tf.keras.utils.set_random_seed(123)

data_dir = r'C:\Users\AI\Desktop\DWA\Train'
data_dir1 = r'C:\Users\AI\Desktop\DWA\Validation'
data_dir2 = r'C:\Users\AI\Desktop\ext1'
data_dir3 = r'C:\Users\AI\Desktop\ext2'
data_dir4 = r'C:\Users\AI\Desktop\ext3'
data_dir5 = r'C:\Users\AI\Desktop\ext4'

batch_size = 16
img_height = 300
img_width = 300

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir1, seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir2, seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds1 = tf.keras.utils.image_dataset_from_directory(
  data_dir3, seed=123, shuffle=False,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds2 = tf.keras.utils.image_dataset_from_directory(
  data_dir4, seed=123, shuffle=False,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds3 = tf.keras.utils.image_dataset_from_directory(
  data_dir5, seed=123, shuffle=False,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

image_extensions = [".png", ".jpg"]  # add there all your images file extensions
img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
for filepath in Path(data_dir1).rglob("*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        if img_type is None:
            print(f"{filepath} is not an image")
        elif img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by TensorFlow")

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_dataset = val_ds.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_ds.prefetch(buffer_size=AUTOTUNE)
test_dataset1 = test_ds1.prefetch(buffer_size=AUTOTUNE)
test_dataset2 = test_ds2.prefetch(buffer_size=AUTOTUNE)
test_dataset3 = test_ds3.prefetch(buffer_size=AUTOTUNE)

preprocess_input = tf.keras.applications.efficientnet_v2.preprocess_input

input_shape = (300, 300, 3)
base_model = tf.keras.applications.EfficientNetB2(input_shape = input_shape,
                                               include_top=False,
                                            weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(300, 300, 3))
x = preprocess_input(inputs)
x = base_model(x, training = False)
x = global_average_layer(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['AUC', 'Recall', 'Accuracy', 'Precision', tf.keras.metrics.SpecificityAtSensitivity(0.85)])
model.summary()
len(model.trainable_variables)

epochs = 4
class_weight = {0: 0.603,
                1: 2.919}
earlystop=tf.keras.callbacks.EarlyStopping('val_loss', restore_best_weights=True, patience=10, verbose=1)
history = model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=validation_dataset, callbacks=[earlystop], class_weight = class_weight)

base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
   layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate = 0.00001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['AUC', 'Recall', 'Accuracy', 'Precision', tf.keras.metrics.SpecificityAtSensitivity(0.85)])

model.summary()

epoch = 1 
earlystop1=tf.keras.callbacks.EarlyStopping('val_loss', restore_best_weights=True, patience=10, verbose=1)

history_fine = model.fit(train_dataset,
                    epochs=epoch,
                    validation_data=validation_dataset, callbacks=[earlystop1], class_weight=class_weight)

x=model.evaluate(test_dataset)
a=model.evaluate(test_dataset1)
b=model.evaluate(test_dataset2)
c=model.evaluate(test_dataset3)

acc = history.history['auc']
val_acc = history.history['val_auc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training AUC')
plt.plot(val_acc, label='Validation AUC')
plt.legend(loc='lower right')
plt.ylabel('AUC')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation AUC')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

