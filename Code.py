#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-success">  
#     <h1></h1>
#      <h1></h1>
# <h1><center><strong>Blindness Detection using Transfer Learning</strong></center></h1>
#      <h1></h1>
#      <h1></h1>
#         
# </div>

# ![image](https://onesight.org/app/uploads/2021/11/OS19113_OneDay_Cover-Images-11191919.jpg)

# <div class="alert alert-block alert-danger">  
# <h2><center><strong>Importing Python Libraries 📕 📗 📘 📙</strong></center></h2>
#         
# </div>

# In[1]:


import numpy as np
import pandas as pd
import cv2
import sys
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import EfficientNetB1
from keras.applications import ResNet152
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.activations import softmax
from keras.activations import elu
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from prettytable import PrettyTable
from sklearn.model_selection import train_test_split
import tensorflow
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import load_img
from tensorflow.keras.utils import load_img
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array


# In[2]:


SEED = 7
np.random.seed(SEED)
tensorflow.random.set_seed(SEED)
dir_path = "data"
IMG_DIM = 299  # 224 399 #
BATCH_SIZE = 12
CHANNEL_SIZE = 3
NUM_EPOCHS = 60
TRAIN_DIR = 'train_images'
TEST_DIR = 'test_images'
FREEZE_LAYERS = 2  
CLASSS = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}


# <div class="alert alert-block alert-danger">  
# <h2><center><strong>Loading the data 📁 📂</strong></center></h2>
#         
# </div>

# In[3]:


df_train = pd.read_csv("D:/PROJECT/FinalProject/train.csv")
df_test = pd.read_csv("D:/PROJECT/FinalProject/test.csv")
NUM_CLASSES = df_train['diagnosis'].nunique()


# <div class="alert alert-block alert-danger">  
# <h2><center><strong>Exploratory data analysis 🔎 📊</strong></center></h2>
#         
# </div>

# In[4]:


print("Training set has {} samples and {} classes.".format(df_train.shape[0], df_train.shape[1]))
print("Testing set has {} samples and {} classes.".format(df_test.shape[0], df_test.shape[1]))


# In[5]:


# Plot pie chart
labels = 'Train', 'Test'
sizes = len(df_train), len(df_test)

fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')

plt.title('Train and Test sets')
plt.show()


# ## Analyze Train Set Labels

# Stages Of Diabetic Retinopathy
# - NO DR
# - Mild
# - Moderate 
# - Servere
# - Proliferative DR

# In[6]:


# Plot pie chart
labels = 'No DR', 'Moderate', 'Mild', 'Proliferative DR', 'Severe'
sizes = df_train.diagnosis.value_counts()

fig1, ax1 = plt.subplots(figsize=(10,7))
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
ax1.axis('equal')

plt.title('Diabetic retinopathy condition labels')
plt.show()


# <div class="alert alert-block alert-info">  
# <h2><center><strong>Split dataset into training and testing</strong></center></h2>
#         
# </div>

# In[7]:


x_train, X_test, y_train, y_test = train_test_split(df_train.id_code, df_train.diagnosis, test_size=0.2,
                                                    random_state=SEED, stratify=df_train.diagnosis)


# <div class="alert alert-block alert-info">  
# <h2><center><strong>Images data visualziation of different classes</strong></center></h2>
#         
# </div>

# In[8]:


def draw_img(imgs, target_dir, class_label='0'):
    fig, axis = plt.subplots(2, 6, figsize=(15, 6))
    for idnx, (idx, row) in enumerate(imgs.iterrows()):
        imgPath = f"D:/PROJECT/FinalProject/{target_dir}/{row['id_code']}.png"
        img = cv2.imread(imgPath)
        row = idnx // 6
        col = idnx % 6
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axis[row, col].imshow(img)
    plt.suptitle(class_label)
    plt.show()


# In[9]:


CLASS_ID = 0
draw_img(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_images', CLASSS[CLASS_ID])


# In[10]:


CLASS_ID = 1
draw_img(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_images', CLASSS[CLASS_ID])


# In[11]:


CLASS_ID = 2
draw_img(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_images', CLASSS[CLASS_ID])


# In[12]:


CLASS_ID = 3
draw_img(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_images', CLASSS[CLASS_ID])


# In[13]:


CLASS_ID = 4
draw_img(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_images', CLASSS[CLASS_ID])


# In[14]:


CLASS_ID = 'Test DataSet'
draw_img(df_test.sample(12, random_state=SEED), 'test_images', CLASS_ID)


# - In Test data, there are some image are bigger and some are having black area. So, testing images also require doing image pre-processing.  
# - May be would be require creating our image Generator.

# <div class="alert alert-block alert-info">  
# <h2><center><strong>Max Min Height and Width</strong></center></h2>
#         
# </div>

# In[15]:


def check_max_min_img_height_width(df, img_dir):
    max_Height , max_Width =0 ,0
    min_Height , min_Width =sys.maxsize ,sys.maxsize 
    for idx, row in df.iterrows():
        imgPath=f"D:/PROJECT/FinalProject/{img_dir}/{row['id_code']}.png"
        img=cv2.imread(imgPath)
        H,W=img.shape[:2]
        max_Height=max(H,max_Height)
        max_Width =max(W,max_Width)
        min_Height=min(H,min_Height)
        min_Width =min(W,min_Width)
    return max_Height, max_Width, min_Height, min_Width


# In[16]:


check_max_min_img_height_width(df_train, TRAIN_DIR)


# In[17]:


check_max_min_img_height_width(df_test, TEST_DIR)


# <div class="alert alert-block alert-info">  
# <h2><center><strong>GrayScale Images</strong></center></h2>
#         
# </div>

# 
# Converting the Ratina Images into Grayscale. So, we can usnderstand the regin or intest .

# ## Image Cropping
# Some images has big blank space. they will take only computation power and add noise to model.
# So better will will crop the blank spaces from images. 

# In[18]:


def draw_img_light(imgs, target_dir, class_label='0'):
    fig, axis = plt.subplots(2, 6, figsize=(15, 6))
    for idnx, (idx, row) in enumerate(imgs.iterrows()):
        imgPath = f"D:/PROJECT/FinalProject/{target_dir}/{row['id_code']}.png"
        img = cv2.imread(imgPath)
        row = idnx // 6
        col = idnx % 6
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_DIM, IMG_DIM))
        img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , IMG_DIM/10) ,-4 ,128) # the trick is to add this line
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        axis[row, col].imshow(img, cmap='gray')
    plt.suptitle(class_label)
    plt.show()


# In[19]:


CLASS_ID = 3
draw_img_light(df_train[df_train.diagnosis == CLASS_ID].head(12), 'train_images', CLASSS[CLASS_ID])


# In[20]:


def crop_image1(img,tol=7):
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]


# In[21]:


def crop_image_from_gray(img,tol=7):
    if img.ndim== 2:
        mask=img>tol
    elif img.ndim==3:
        gray_img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        mask=gray_img>tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if check_shape ==0: 
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            print(img1.shape,img2.shape,img3.shape)            
            img=np.stack([img1,img2,img3],axis=1)
            print(img.shape)
            return img


# In[22]:


def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_DIM, IMG_DIM))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
        
    return image


# In[23]:


def crop_image(img,tol=7):
    w, h = img.shape[1],img.shape[0]
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.blur(gray_img,(5,5))
    shape = gray_img.shape 
    gray_img = gray_img.reshape(-1,1)
    quant = quantile_transform(gray_img, n_quantiles=256, random_state=0, copy=True)
    quant = (quant*256).astype(int)
    gray_img = quant.reshape(shape)
    xp = (gray_img.mean(axis=0)>tol)
    yp = (gray_img.mean(axis=1)>tol)
    x1, x2 = np.argmax(xp), w-np.argmax(np.flip(xp))
    y1, y2 = np.argmax(yp), h-np.argmax(np.flip(yp))
    if x1 >= x2 or y1 >= y2 : # something wrong with the crop
        return img # return original image
    else:
        img1=img[y1:y2,x1:x2,0]
        img2=img[y1:y2,x1:x2,1]
        img3=img[y1:y2,x1:x2,2]
        img = np.stack([img1,img2,img3],axis=-1)
    return img

def process_image(image, size=512):
    image = cv2.resize(image, (size,int(size*image.shape[0]/image.shape[1])))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        image = crop_image(image, tol=15)
    except Exception as e:
        image = image
        print( str(e) )
    return image


# <div class="alert alert-block alert-info">  
# <h2><center><strong>Data Pre-Processing</strong></center></h2>
#         
# </div>

# - Croping Images randomly for resizing.

# In[24]:


def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    img = img[y:(y + dy), x:(x + dx), :]
    return img

def crop_generator(batches, crop_length):
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[0] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)


# - Adding image type with image in dataframe

# In[25]:


df_train.id_code = df_train.id_code.apply(lambda x: x + ".png")
df_test.id_code = df_test.id_code.apply(lambda x: x + ".png")
df_train['diagnosis'] = df_train['diagnosis'].astype('str')


# <div class="alert alert-block alert-info">  
# <h2><center><strong>Image Data Generator</strong></center></h2>
#         
# </div>

# In[26]:


datagenerator=ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=40, zoom_range=0.2, shear_range=0.1,fill_mode='nearest')


# In[27]:


imgPath = f"D:/PROJECT/FinalProject/train_images/cd54d022e37d.png"
# Loading image
img = load_img(imgPath)
data = img_to_array(img)
samples =np.expand_dims(data, 0)
i=5
it=datagenerator.flow(samples , batch_size=1)
for i in range(5):
    plt.subplot(230 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    plt.imshow(image)
plt.show()


# In[28]:


train_datagen = ImageDataGenerator(rescale=1. / 255, 
                                         validation_split=0.15, 
                                         horizontal_flip=True,
                                         vertical_flip=True, 
                                         rotation_range=40, 
                                         zoom_range=0.2, 
                                         shear_range=0.1,
                                        fill_mode='nearest')


# In[29]:


df_train


# In[30]:


train_generator = train_datagen.flow_from_dataframe(dataframe=df_train,
                                                    directory="D:/PROJECT/FinalProject/train_images/",
                                                    x_col="id_code",
                                                    y_col="diagnosis",
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="categorical",
                                                    target_size=(224, 224),
                                                    subset='training',
                                                    shaffle=True,
                                                    seed=SEED,
                                                    )

train_generator
test_D = df_train.sample(n=360)
test_d = train_datagen.flow_from_dataframe(dataframe=test_D,
                                                    directory="D:/PROJECT/FinalProject/train_images/",
                                                    x_col="id_code",
                                                    y_col="diagnosis",
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="categorical",
                                                    target_size=(224, 224),
                                                    subset='validation',
                                                    shaffle=True,
                                                    seed=SEED,
                                                    )

test_d1 = train_datagen.flow_from_dataframe(dataframe=test_D,
                                                    directory="D:/PROJECT/FinalProject/train_images/",
                                                    x_col="id_code",
                                                    #y_col="diagnosis",
                                                    batch_size=BATCH_SIZE,
                                                    class_mode=None,
                                                    target_size=(224, 224),
                                                    #subset='validation',
                                                    shaffle=True,
                                                    seed=SEED,
                                           )

# testp_generator = testp_datagen.flow_from_dataframe(
#     testp_df, 
#     "/content/drive/MyDrive/Reduceddata_Nocategories/Testing Dataset",
#     x_col='filename',
#     y_col=None,
#     class_mode=None,
#     target_size=(180,180),
#     batch_size=batch_size,
#     shuffle=False 
# )



# <div class="alert alert-block alert-success">  
# <h2><center><strong>VGG16 Model using as transfer learning</strong></center></h2>
#         
# </div>

# In[31]:


vgg16_model = VGG16(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

outputs = vgg16_model.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(5, activation="softmax")(outputs)

vgg16_model = Model(inputs=vgg16_model.input, outputs=outputs)

vgg16_model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
)
vgg16_model.summary()


# # Training and validating 

# In[32]:


history= vgg16_model.fit_generator(generator=train_generator,
                                     steps_per_epoch=5,
                                     validation_data=train_generator,
                                     validation_steps=5,
                                     epochs=10)


# ### Training and Validation Accuracy

# In[33]:


print(history.history)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Training and Validation Loss

# In[34]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[35]:


print(X_test)


# ### Testing the Trained model with test data

# In[36]:


accr = vgg16_model.evaluate(test_d)
accr


# ### Accuracy

# In[37]:


y_pred = vgg16_model.predict(test_d1)
y_pred=y_pred.argmax(axis=1)
#print('Test set\n  Accuracy: {:0.5f}'.format(accr[1]))


# ### Precision, Recall, F1

# In[38]:


print('\n')
print("Precision, Recall, F1")
print('\n')
labels =['No DR', 'Moderate', 'Mild', 'Proliferative DR', 'Severe']
CR=classification_report(y_test[0:360], y_pred, target_names=labels)
print(CR)
print('\n')


# ### Confusion matrix

# In[39]:


CM=confusion_matrix(y_test[0:360], y_pred)
fig, ax = plot_confusion_matrix(conf_mat=CM,figsize=(10, 10),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=False)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.show()


# <div class="alert alert-block alert-success">  
# <h2><center><strong>VGG19 Model using as transfer learning</strong></center></h2>
#         
# </div>

# In[40]:


vgg19_model = VGG19(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

outputs = vgg19_model.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(5, activation="softmax")(outputs)

vgg19_model = Model(inputs=vgg19_model.input, outputs=outputs)

vgg19_model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
)
vgg19_model.summary()


# # Training and validating 

# In[42]:


history= vgg19_model.fit_generator(generator=train_generator,
                                     steps_per_epoch=1000,
                                     #validation_data=valid_generator,
                                     validation_steps=1000,
                                     epochs=10)


# ### Training and Validation Accuracy

# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Training and Validation Loss

# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Testing the Trained model with test data

# In[ ]:


accr = vgg19_model.evaluate(X_test,y_test)


# ### Accuracy

# In[ ]:


y_pred = vgg19_model.predict(X_test)
y_pred=y_pred.argmax(axis=1)
print('Test set\n  Accuracy: {:0.5f}'.format(accr[1]))


# ### Precision, Recall, F1

# In[ ]:


print('\n')
print("Precision, Recall, F1")
print('\n')
labels =['No DR', 'Moderate', 'Mild', 'Proliferative DR', 'Severe']
CR=classification_report(y_test, y_pred, target_names=labels)
print(CR)
print('\n')


# ### Confusion matrix

# In[ ]:


CM=confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=CM,figsize=(10, 10),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=False)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.show()


# <div class="alert alert-block alert-success">  
# <h2><center><strong>EfficientNet Model using as transfer learning</strong></center></h2>
#         
# </div>

# In[ ]:


EfficientNet_model = EfficientNetB1(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

outputs = EfficientNet_model.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(5, activation="softmax")(outputs)

model_EfficientNet = Model(inputs=EfficientNet_model.input, outputs=outputs)

model_EfficientNet.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
)
model_EfficientNet.summary()


# # Training and validating 

# In[ ]:


history= model_EfficientNet.fit_generator(generator=train_generator,
                                     steps_per_epoch=1000,
                                     #validation_data=valid_generator,
                                     validation_steps=1000,
                                     epochs=10)


# ### Training and Validation Accuracy

# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Training and Validation Loss

# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Testing the Trained model with test data

# In[ ]:


accr = model_EfficientNet.evaluate(X_test,y_test)


# ### Accuracy

# In[ ]:


y_pred = model_EfficientNet.predict(X_test)
y_pred=y_pred.argmax(axis=1)
print('Test set\n  Accuracy: {:0.5f}'.format(accr[1]))


# ### Precision, Recall, F1

# In[ ]:


print('\n')
print("Precision, Recall, F1")
print('\n')
labels =['No DR', 'Moderate', 'Mild', 'Proliferative DR', 'Severe']
CR=classification_report(y_test, y_pred, target_names=labels)
print(CR)
print('\n')


# ### Confusion matrix

# In[ ]:


CM=confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=CM,figsize=(10, 10),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=False)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.show()


# <div class="alert alert-block alert-success">  
# <h2><center><strong>ResNet152 Model using as transfer learning</strong></center></h2>
#         
# </div>

# In[ ]:


ResNet152Re = ResNet152(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

outputs = ResNet152Re.output
outputs = Flatten(name="flatten")(outputs)
outputs = Dropout(0.5)(outputs)
outputs = Dense(5, activation="softmax")(outputs)

model_ResNet152Net = Model(inputs=ResNet152Re.input, outputs=outputs)

model_ResNet152Net.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy']
)
model_ResNet152Net.summary()


# # Training and validating 

# In[ ]:


history= model_ResNet152Net.fit_generator(generator=train_generator,
                                     steps_per_epoch=1000,
                                     #validation_data=valid_generator,
                                     validation_steps=1000,
                                     epochs=10)


# ### Training and Validation Accuracy

# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Training and Validation Loss

# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Testing the Trained model with test data

# In[ ]:


accr = model_ResNet152Net.evaluate(X_test,y_test)


# ### Accuracy

# In[ ]:


y_pred = model_ResNet152Net.predict(X_test)
y_pred=y_pred.argmax(axis=1)
print('Test set\n  Accuracy: {:0.5f}'.format(accr[1]))


# ### Precision, Recall, F1

# In[ ]:


print('\n')
print("Precision, Recall, F1")
print('\n')
labels =['No DR', 'Moderate', 'Mild', 'Proliferative DR', 'Severe']
CR=classification_report(y_test, y_pred, target_names=labels)
print(CR)
print('\n')


# ### Confusion matrix

# In[ ]:


CM=confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=CM,figsize=(10, 10),
                                show_absolute=True,
                                show_normed=True,
                                colorbar=False)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.show()


# <div class="alert alert-block alert-info">  
# <h2><center><strong>Accuracy Comparison of all models Results</strong></center></h2>
#         
# </div>

# In[ ]:


x = PrettyTable()
print('\n')
print("Comparison of all models results")
x.field_names = ["Model", "Accuracy"]
x.add_row(["VGG16 Model", acc1])
x.add_row(["VGG19 Model", acc2])
x.add_row(["EfficientNet Model", acc3])
x.add_row(["Resnet152 Model", acc4])
print(x)
print('\n')


# In[ ]:


aa=pd.DataFrame()

aa['VGG16 Model']=[acc1]
aa['VGG19 Model']=[acc2]
aa['EfficientNet Model']=[acc3]
aa['Resnet152 Model']=[acc4]
aa=aa.T


colors_list = ['#5cb85c']
result_pct = aa

ax = result_pct.plot(kind='bar',figsize=(15,6),width = 0.8,color = colors_list,edgecolor=None)
plt.legend(labels=aa.columns,fontsize= 14)
plt.title("Comparison of all algorithms Results",fontsize= 20)
print('\n')
print('\n')
plt.xticks(fontsize=14)
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.yticks([])

plt.legend('',frameon=False)
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy() 
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center')


# > EfficientNet model performed well and gave the highest accuracy. 
