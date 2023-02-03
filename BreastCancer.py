import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import pathlib
from skimage.io import imread, imshow
from PIL import Image as im
from sklearn import preprocessing
import glob
import shutil

le = preprocessing.LabelEncoder()                   #Pre-Prcoessing Step. To move all images of sub-folders into two subfolders(Malignant and Benign)
data_dir = "BreastImages"
directory = "MalignantData"
parant_directory =os.path.join (r"C:\Users\Nasir\PycharmProjects")
make_directory = directory
os.mkdir(make_directory)
new_directory=make_directory

def move(shift):                                    #Function to move all the images 
   for images in glob.iglob(f'{shift}/*'):

        if (images.endswith(".png")):
            shutil.move(images, new_directory)

def Recurseion(Record):
    curr = len(Record) - 1
    print('the curr is:',curr)
    current = Record[curr]
    print('The main direectory is:', current)
    filelist = pathlib.Path(current)
    filelist = os.listdir(current)
    print('The filelist is: ', filelist)


    lps = len(filelist)
    i=0
    while i<lps:
        sub = os.path.join(current, filelist[i])
        sub= pathlib.Path(sub)
        print('The bc or subdirectory is  :', sub)
        a=os.listdir(sub)
        f1=a[0]
        file_name, file_extension = os.path.splitext(f1)
        if file_extension=='.png':                           # Check weather the sub-folder is an image or image
            move(sub)
        elif file_extension=='':
            print('Is this even repeated?')
            Record.append(sub)
            print('the new record is:',Record)
            Recurseion(Record)
        i+=1

Record=[]

dir =os.path.join(r"C:\Users\Nasir\PycharmProjects\my\breast\malignant\SOB")  # Main directory where all the subfolders are placed initially
Record.append(dir)

Recurseion(Record)


batch_size = 32
img_height = 128
img_width = 128
epochs=40

class_names = os.listdir(data_dir)
print(class_names)
classlength=len(class_names)
img_channels=3


trainimagearray=[]                #Pre-Prcoessing step to move all the images to np.arrays
trainlabearray=[]
testimagearray=[]
testlabelarray=[]

for c in range(classlength):
 current_class=class_names[c]
 clss=os.path.join(data_dir,current_class)
 lst=os.listdir(clss)
 total=(len(lst))
 testnumbers=(int(len(lst)*0.2))
 trainnumbers=total-testnumbers

 for i in range(trainnumbers):
    img=os.path.join(clss,lst[i])
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    img=img/255.0
    img = cv2.resize(img, (img_height, img_width))
    trainimagearray.append(np.array(img))
    trainlabearray.append(current_class)
 for t in range(testnumbers):
    img2=os.path.join(clss,lst[t+trainnumbers])
    img2 = cv2.imread(img2, cv2.IMREAD_COLOR)
    img2=img2/255.0
    img2 = cv2.resize(img2, (img_height, img_width))
    testimagearray.append(np.array(img2))
    testlabelarray.append(current_class)

trainimg= np.zeros((len(trainimagearray), img_height, img_width, img_channels))
trainlabel=np.zeros(len(trainlabearray))
testimage=np.zeros((len(testimagearray), img_height, img_width, img_channels))
testlabel=np.zeros(len(testlabelarray))

a=trainlabearray                    # Convert all training labels into numbers.
le.fit(a)
le.classes_
encodedlabel = le.transform(a)

a2=testlabelarray
le.fit(a2)
le.classes_
encodedtestlabel = le.transform(a2)

for i in range(len(trainimagearray)):
    trainimg[i]=trainimagearray[i]
    trainlabel[i]=encodedlabel[i]
for t in range(len(testimagearray)):
    testimage[t]=testimagearray[t]
    testlabel[t]=encodedtestlabel[t]


#print('The shape of training:', trainimg.shape)
print('The  shape of training images are :', trainimg.shape)
print('The  shape of test images are :',testimage.shape)

#plt.imshow(testimage[6], interpolation='nearest')
#plt.show()


inputs = tf.keras.layers.Input((img_height, img_width, img_channels))


c1 = tf.keras.layers.Conv2D(16, 5, activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.4)(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, 5, activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.4)(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, 5, activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.4)(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)


p4=tf.keras.layers.Flatten()(p3)
p5=tf.keras.layers.Dense(256, activation='relu')(p4)

p6 = tf.keras.layers.Dense(128, activation='relu')(p5)
p6 = tf.keras.layers.Dropout(0.4)(p6)

p7 = tf.keras.layers.Dense(64, activation='relu')(p6)
p7 = tf.keras.layers.Dropout(0.4)(p7)

outputs = tf.keras.layers.Dense(classlength, activation='softmax')(p7)


model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='accuracy'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

history= model.fit(trainimg, trainlabel, batch_size=batch_size, epochs=epochs,validation_data=(testimage, testlabel))

predictions = model.predict(testimage)


test_img_number = np.random.randint(0, len(testimagearray))
test_loss, test_acc = model.evaluate(testimage,  testlabel, verbose=1)
print("The test accuracy is:", test_acc)
score = tf.nn.softmax(predictions[test_img_number])
print('This is the predicted flower',class_names[np.argmax(score)])
print("This is the actual test value",testlabelarray[test_img_number])



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
