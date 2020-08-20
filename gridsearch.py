import pandas as pd
import gc
import os
import matplotlib.pyplot as plt
import random
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from time import time
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

directory = "dataset"

print(os.listdir(directory))

IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

filenames = os.listdir(directory)
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
df.head()

df['category'].value_counts().plot.bar()

sample = random.choice(filenames)
image = load_img(directory + "/" + sample)
plt.imshow(image)


def define_model(
        nodes=16,
        loss='binary_crossentropy',
        opt="rmsprop",
        pooling=None,
        dropout=False,
        fc=1,
        act=tf.nn.relu,
        blocks=0
):

    inputs = tf.keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

    x = Conv2D(64, (3, 3), activation=act, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))(inputs)

    for _ in range(0, blocks):
        x = Conv2D(64, (3, 3), padding="same", activation=act)(x)
        # if pooling == "MAX":
        x = MaxPooling2D(pool_size=(5, 5))(x)
        # if pooling == "AVG":
        #    x = AveragePooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    for _ in range(0, fc):
        x = Dense(nodes, activation=act)(x)

    if dropout:
        x = Dropout(0.2)(x)

    outputs = Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    print(model.summary())

    return model


earlystop = EarlyStopping(patience=3)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', verbose=1, factor=0.5, min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

train_df, test_df = train_test_split(df, test_size=0.10, random_state=42)
train_df['category'].value_counts().plot.bar()
test_df['category'].value_counts().plot.bar()
plt.show()

total_train = train_df.shape[0]
# total_validate = validate_df.shape[0]
batch_size = 5

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df,
    directory,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)

plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i + 1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

# K-Fold CV

# Parameters for grid search

n_folds = 3  # Number of folds for K-fold CV
epochs = 8  # Number for model training epochs
batch = 5  # Batch size of the training
blocks = [1, 2, 3]  # Number of blocks (Conv2D + Pooling layers) in the model, by default 2
nodes = [128, 256, 512]  # List of the number of nodes for the gridseach
FC_layers = [1, 2, 3]  # List of the number of fully connected layers after Flatten in the model
acts = ["sigmoid", "tanh", "relu"]  # List of activation functions used in the model
# poolings = ["MAX", "AVG"]  # List of poolings used in the model
opts = ["adam", "sgd", "rmsprop"]  # List of optimizers used in the model
# structure = "cnn"

information = True  # If True print completed model parameters and average accuracy for folds

# prepare cross validation
kfold = KFold(n_folds, shuffle=True, random_state=1)
splits = kfold.split(train_df)

# Prepare DataFrame for results saving
Results = pd.DataFrame(columns=["Optimizer", "Fully Connected", "Activation", "Blocks", "Node", "AVG Train",
                                "AVG Valid", "AVG Test", "Time"])

for o in opts:
    # for fc in FC_layers:
    # for a in acts:
    # for b in blocks:
    for n in nodes:

        start = time()
        avg_train_acc = 0
        avg_val_acc = 0
        avg_test_acc = 0

        print("*******************************************************************")
        print(str(o) + " " + str(n))
        print("*******************************************************************")

        for train_ix, test_ix in kfold.split(train_df):
            model = define_model(nodes=n, opt=o, blocks=1, fc=2, act='relu')

            train_generator = train_datagen.flow_from_dataframe(
                train_df.iloc[train_ix],
                directory,
                x_col='filename',
                y_col='category',
                target_size=IMAGE_SIZE,
                class_mode='categorical',
                batch_size=batch_size
            )

            validation_generator = train_datagen.flow_from_dataframe(
                train_df.iloc[test_ix],
                directory,
                x_col='filename',
                y_col='category',
                target_size=IMAGE_SIZE,
                class_mode='categorical',
                batch_size=batch_size
            )

            history = model.fit(
                train_generator,
                verbose=2,
                epochs=epochs,
                validation_data=validation_generator,
                callbacks=callbacks
            )

            _, acc = model.evaluate(train_generator)
            print(f"Training Acc: {acc}")
            avg_train_acc = avg_train_acc + acc

            _, acc = model.evaluate(validation_generator)
            print(f"Validation Acc: {acc}")
            avg_val_acc = avg_val_acc + acc

            _, acc = model.evaluate(test_generator)
            print(f"Test Acc: {acc}")
            avg_test_acc = avg_test_acc + acc

            K.clear_session()
            gc.collect()

        Results = Results.append({"Optimizer": o,
                                  "Fully Connected": 2,
                                  "Activation": 'relu',
                                  "Blocks": 1,
                                  "Node": n,
                                  "AVG Train": avg_train_acc / n_folds,
                                  "AVG Valid": avg_val_acc / n_folds,
                                  "AVG Test": avg_test_acc / n_folds,
                                  "Time": time() - start},
                                 ignore_index=True)

# # Final model with confusion matrix and etc.

# model = define_model(structure)
#
# train_generator = train_datagen.flow_from_dataframe(
#     train_df.iloc[train_ix],
#     directory,
#     x_col='filename',
#     y_col='category',
#     target_size=IMAGE_SIZE,
#     class_mode='categorical',
#     batch_size=batch_size
#     )
#
# history = model.fit(
#     train_generator,
#     epochs=epochs,
#     steps_per_epoch=total_train//batch_size,
#     callbacks=callbacks
#     )


Results.to_csv("results/cv_only_model.csv")

# predicting images
# img = keras.preprocessing.image.load_img('IMAGE_PATH', target_size=(128, 128))
# x = keras.preprocessing.image.img_to_array(img)
# x = np.expand_dims(x, axis=0)

# images = np.vstack([x])
# classes = model.predict(images, batch_size=10)
# print("Probabilities:",classes)

# if (np.argmax(classes,axis=1)[0] == 1):
#    print("DOG!")
# if (np.argmax(classes,axis=1)[0] == 0):
#    print("CAT!")

# plot_model(model, to_file='cnn.png', show_shapes=True, show_layer_names=True)






