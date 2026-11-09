import os

# i'm setting tensorflow to only show errors, hiding all the info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
# i'm also suppressing the c++ level logs that tensorflow spits out
tf.get_logger().setLevel('ERROR')

# setting up my directory paths where i'll store images and save my models
image_data_dir = "../Image Data"
project_dir = "."
models_dir = os.path.join(project_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# this is where my actual dataset lives - the final plantvillage folder
file_path = os.path.join(image_data_dir, "Final PlantVillage")

# checking if the folder actually exists, if not i'll throw an error
if not os.path.isdir(file_path):
    raise FileNotFoundError(
        f"dataset directory not found: {file_path}\n"
        "please ensure the 'final plantvillage' folder exists in the image data directory."
    )

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# these are my hyperparameters - batch size, image dimensions, and how many epochs i want
batch_size = 32
image_size = 256
channels = 3
epochs = 50

import os
import shutil
import random

# i'm setting up where to read images from and where to save the balanced dataset
source_dir = file_path
working_dir = os.path.join(image_data_dir, "balanced_dataset")

# i only want classes with at least 900 images, and i'll cap each at 1200 max
min_limit = 900
max_limit = 1200

# creating the working directory if it doesn't exist
if not os.path.exists(working_dir):
    os.makedirs(working_dir)

# going through each class folder and balancing the dataset
for class_folder in os.listdir(source_dir):
    src_path = os.path.join(source_dir, class_folder)
    
    if os.path.isdir(src_path):
        images = os.listdir(src_path)
        count = len(images)
        
        # skipping classes that don't have enough images
        if count < min_limit:
            print(f"skipping {class_folder}: only {count} images (too low).")
            continue
            
        # figuring out how many images to keep, max 1200
        keep_count = min(count, max_limit)
        selected_images = random.sample(images, keep_count)
        
        # creating the destination folder and copying my selected images
        dest_path = os.path.join(working_dir, class_folder)
        os.makedirs(dest_path, exist_ok=True)
        
        for img in selected_images:
            shutil.copy(os.path.join(src_path, img), os.path.join(dest_path, img))
            
        print(f"saved {keep_count} images for {class_folder}")

# loading my balanced dataset into tensorflow, resizing images and creating batches
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    working_dir,
    shuffle = True,
    image_size = (image_size, image_size),
    batch_size = batch_size
)

# getting the class names from the dataset
class_names = dataset.class_names
print(f"the class names are {class_names}")

# counting how many images i have per class
counts = {class_name: 0 for class_name in class_names}

# looping through the dataset to count images
for images, labels in dataset:
    for label in labels.numpy():
        counts[class_names[label]] += 1

# printing out my counts
for class_name, count in counts.items():
    print(f"{class_name}: {count} images")

# making a bar chart to visualize how many images i have per class
plt.figure(figsize=(10, 6))
plt.bar(counts.keys(), counts.values(), color="skyblue")
plt.xticks(rotation=45)
plt.ylabel("number of images")
plt.title("images per class")
plt.show()


print(f" length of the dataset = {len(dataset)}")

# showing a sample of 12 images from my dataset to see what i'm working with
plt.figure(figsize = (10, 10))
for image_batch, label_batch in dataset.take(1):
  for i in range(12):
    ax = plt.subplot(3, 4, i+1)
    plt.imshow(image_batch[i].numpy().astype('uint8'))
    plt.title(class_names[label_batch[i]])
    plt.axis('off')

# function to split my dataset into train, validation, and test sets
def split_dataset(ds, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=True, shuffle_buffer=10000):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ds_len = len(ds)
        if shuffle:
                ds = ds.shuffle(shuffle_buffer, seed=42)
        train_len = int(train_ratio * ds_len)
        val_len = int(val_ratio * ds_len)
        train_set = ds.take(train_len)
        val_set = ds.skip(train_len).take(val_len)
        test_set = ds.skip(train_len + val_len)
        return train_set, val_set, test_set

# splitting my dataset into train, validation, and test
train_ds, val_ds, test_ds = split_dataset(dataset)
print(f"the length of the train dataset = {len(train_ds)}")
print(f"the length of the test dataset = {len(test_ds)}")
print(f"the length of the validation dataset = {len(val_ds)}")

# optimizing my datasets with shuffling and prefetching for faster training
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# creating a preprocessing layer to resize and normalize my images
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(image_size, image_size),
    layers.Rescaling(1.0/255)
])

# creating data augmentation to flip and rotate images for better training
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2)
])


# applying data augmentation to my training set
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training = True), y)).prefetch(buffer_size = tf.data.AUTOTUNE)

# setting up my model input shape and number of classes
input_shape = (batch_size, image_size, image_size, channels)
n_classes = len(class_names)

# building my cnn model layer by layer
model = models.Sequential([
    resize_and_rescale,

    # first conv layer finds basic edges in the images
    layers.Conv2D(filters = 32, kernel_size = (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.SpatialDropout2D(0.2),

    # second conv layer finds simple shapes
    layers.Conv2D(filters = 64, kernel_size = (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.SpatialDropout2D(0.2),

    # third conv layer finds complex patterns like leaf spots and veins
    layers.Conv2D(filters = 128, kernel_size = (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.SpatialDropout2D(0.2),

    # fourth conv layer finds high-level disease features
    layers.Conv2D(filters = 256, kernel_size = (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),

    # dense layer that interprets all the features i found
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    # final output layer that gives me probabilities for each class
    layers.Dense(n_classes, activation='softmax')
])

# building the model with my input shape
model.build(input_shape=(None, image_size, image_size, channels))

model.summary()


# compiling my model with adam optimizer and sparse categorical crossentropy loss
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
)

# setting up early stopping so training stops if validation loss doesn't improve for 5 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5,
    restore_best_weights=True
)

# reducing learning rate if validation loss plateaus
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,
    patience=3,
    min_lr=0.00001
)

# saving my best model checkpoint based on validation accuracy
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(models_dir, 'plant_model_best.keras'),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# training my model with all my callbacks
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[early_stopping, reduce_lr, checkpoint_callback] 
)

# saving my final trained model
model.save(os.path.join(models_dir, 'final_model.keras'))

print(f"model saved successfully in {models_dir}!")

import json

# saving my class names to a json file so i remember the order
with open(os.path.join(models_dir, 'class_names.json'), 'w') as f:
    json.dump(class_names, f)

import pandas as pd

# saving my training history to a csv file
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(models_dir, 'training_history.csv'), index=False)

import matplotlib.pyplot as plt

# plotting my training and validation accuracy over time
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('model accuracy')
plt.legend()

# plotting my training and validation loss over time
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('model loss')
plt.legend()

plt.savefig('./training_history_graph.png', dpi=300, bbox_inches='tight')
plt.show()

# evaluating my model on the test set
scores = model.evaluate(test_ds)
print(f"scores:\ntesting loss = {scores[0]}\ntesting accuracy = {scores[1]}")

# function to predict the class of a single image
def predict_class(model, img):
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.expand_dims(arr, 0)
    preds = model.predict(arr)
    pred_label = class_names[np.argmax(preds[0])]
    conf = round(100 * np.max(preds[0]), 2)
    return pred_label, conf



# showing predictions on 9 test images with actual vs predicted labels
plt.figure(figsize=(18, 18))
for imgs, lbls in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(imgs[i].numpy().astype("uint8"))
        actual = class_names[lbls[i].numpy()]
        img_pred = tf.expand_dims(imgs[i], 0)
        pred_probs = model.predict(img_pred, verbose=0)
        pred_idx = np.argmax(pred_probs[0])
        pred_label = class_names[pred_idx]
        conf = 100 * np.max(pred_probs[0])
        color = "green" if pred_label == actual else "red"
        ax.set_title(
            f"actual: {actual}\npred: {pred_label}\nconfidence: {conf:.2f}%",
            color=color, fontsize=10, fontweight='bold')
        ax.axis("off")
# adjusting spacing so titles don't overlap
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.savefig('./inference_result.png', dpi=300, bbox_inches='tight')
plt.show()


# function to plot a confusion matrix showing how well my model predicts each class
def plot_confusion_matrix(model, dataset, class_names):
    # collecting all my predictions and true labels
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    # computing the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # plotting it as a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.title("confusion matrix")
    plt.savefig(os.path.join(project_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

plot_confusion_matrix(model, test_ds, class_names)


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# collecting all predictions and labels from my test set
all_labels = []
all_preds = []

print("evaluating model on test dataset...")
for images, labels in test_ds:
    # getting probability predictions from my model
    preds = model.predict(images, verbose=0)
    
    # picking the class with the highest probability
    predicted_indices = np.argmax(preds, axis=1)
    
    all_labels.extend(labels.numpy())
    all_preds.extend(predicted_indices)

# generating a detailed classification report
report = classification_report(all_labels, all_preds, target_names=class_names)
print("\nclassification report:\n")
print(report)

# saving the report to a text file
with open(os.path.join(models_dir, "classification_report.txt"), "w") as f:
    f.write(report)

print("report written successfully!!")

import shutil
# zipping all my model files into one archive
output_zip = os.path.join(project_dir, 'project_files.zip')
shutil.make_archive(os.path.splitext(output_zip)[0], 'zip', models_dir)
print(f"model files archived to: {output_zip}")

# trying to create a file link if i'm in jupyter, otherwise just skipping it
try:
    from IPython.display import FileLink
    FileLink(r'project_files.zip')
except ImportError:
    pass
