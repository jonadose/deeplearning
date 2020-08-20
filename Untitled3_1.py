#!/usr/bin/env python
# coding: utf-8

# # Cats VS Dogs Classification

# ## Libraries import

# In[52]:


import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from tensorflow.keras import backend as K
import gc
from sklearn.model_selection import KFold
from time import time
from sklearn.metrics import classification_report, confusion_matrix

directory = "../Datasets/SDUCatsVsDogs"

print(os.listdir(directory))


# ## Parameters

# In[3]:


IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


# In[4]:


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


# In[5]:


df['category'].value_counts().plot.bar()


# In[6]:


sample = random.choice(filenames)
image = load_img(directory+"/"+sample)
plt.imshow(image)


# In[59]:


#MODEL1

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, Add
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def define_model(structure):
    
    if structure == "cnn":  
        inputs = keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

        x = Conv2D(64, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(512, activation='relu')(x)

        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        x = outputs = Dense(2, activation='softmax')(x) # 2 because we have cat and dog classes

        model = keras.Model(inputs, outputs)

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        model.summary()
    
    if structure == "resnet": 
        inputs = keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    
        x = Conv2D(64, (7, 7), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))(inputs)

        x_temp = x
        x = Conv2D(64, (3, 3), padding = "same",  activation='relu')(x)
        x = Conv2D(64, (3, 3), padding = "same",  activation='relu')(x)
        x = Add()([x, x_temp])
        x = Conv2D(64, (3, 3), padding = "same",  activation='relu')(x)
        x = Conv2D(64, (3, 3), padding = "same",  activation='relu')(x)
        x = Add()([x, x_temp])
        x = Conv2D(64, (3, 3), padding = "same",  activation='relu')(x)
        x = Conv2D(64, (3, 3), padding = "same",  activation='relu')(x)
        x = Add()([x, x_temp])
        x = Conv2D(64, (3, 3), padding = "same",  activation='relu')(x)
        x = Conv2D(64, (3, 3), padding = "same",  activation='relu')(x)

        x = MaxPooling2D(pool_size=(5, 5))(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        outputs = Dense(2, activation='softmax')(x) # 2 because we have cat and dog classes

        model = keras.Model(inputs, outputs)

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])      
    
    return model


# In[8]:


earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]


# In[9]:


df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 


# ## Train Test split

# In[40]:


train_df, test_df = train_test_split(df, test_size=0.10, random_state=42)


# In[41]:


train_df['category'].value_counts().plot.bar()


# In[42]:


test_df['category'].value_counts().plot.bar()


# In[13]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=5


# In[43]:


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    directory, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    directory, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[44]:


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    directory, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)


# In[45]:


plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# In[205]:


epochs=3 if FAST_RUN else 50

model = define_model("cnn")

history = model.fit(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# ## K-Fold CV

# In[60]:


# Parameters for grid search

n_folds = 5 # Number of folds for K-fold CV
epochs = 5 # Number for model training epochs
batch = 32 # Batch size of the training
blocks = [1,2,3] # Number of blocks (Conv2D + Pooling layers) in the model, by default 2
nodes = [64,128,256] # List of the number of nodes for the gridseach
FC_layers = [1,2,3] # List of the number of fully connected layers after Flatten in the model
acts = ["sigmoid", "tanh", "relu"] # List of activation functions used in the model
poolings = ["MAX","AVG"] # List of poolings used in the model
opts = ["adam","sgd","rmsprop"] # List of optimizers used in the model
epochs=3
structure = "cnn"


information = True # If True print completed model parameters and average accuracy for folds

# prepare cross validation
kfold = KFold(n_folds, shuffle=True, random_state=1)
splits = kfold.split(train_df)


# Prepare DataFrame for results saving
Results = pd.DataFrame(columns=["Structure",
                                "AVG Train",
                                "AVG Valid",
                                "AVG Test",
                               "Time"])

start = time()
avg_train_acc = 0
avg_val_acc = 0
avg_test_acc = 0

for train_ix, test_ix in kfold.split(train_df):
    
    model = define_model(structure)
    
    train_generator = train_datagen.flow_from_dataframe(
    train_df.iloc[train_ix], 
    directory, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
    )
    
    validation_generator = validation_datagen.flow_from_dataframe(
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
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate//batch_size,
        steps_per_epoch=total_train//batch_size,
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
    
Results = Results.append({"Structure": structure,
                          "AVG Train": avg_train_acc/n_folds,
                          "AVG Valid": avg_val_acc/n_folds,
                          "AVG Test": avg_test_acc/n_folds,
                         "Time": time() - start}, ignore_index=True)


# In[61]:


Results


# # Final model with confussion matrix and etc.

# In[66]:



model = define_model(structure)
    
train_generator = train_datagen.flow_from_dataframe(
train_df.iloc[train_ix], 
directory, 
x_col='filename',
y_col='category',
target_size=IMAGE_SIZE,
class_mode='categorical',
batch_size=batch_size
)

validation_generator = validation_datagen.flow_from_dataframe(
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
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# In[69]:


Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['Cats', 'Dogs']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))


# In[72]:


import seaborn as sns

data = confusion_matrix(validation_generator.classes, y_pred)
df_cm = pd.DataFrame(data, columns=np.unique(validation_generator.classes), index = np.unique(validation_generator.classes))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size


# In[163]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# In[186]:


# predicting images
img = keras.preprocessing.image.load_img('D:/GitHub Repositories/Datasets/CatsVSDogs/test1/8.jpg', target_size=(128, 128))
x = keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print("Probabilities:",classes)

if (np.argmax(classes,axis=1)[0] == 1):
    print("DOG!")
if (np.argmax(classes,axis=1)[0] == 0):
    print("CAT!")


# In[77]:


history.history


# In[78]:


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='rnn.png' ,show_shapes=True, show_layer_names=True)


# In[ ]:




