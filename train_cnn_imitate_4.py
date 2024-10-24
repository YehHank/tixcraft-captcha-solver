from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from PIL import Image
import numpy as np
import csv

# 常量定義
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 120
DROPOUT_RATE = 0.3
BATCH_SIZE = 64
EPOCHS = 100
LETTERSTR = "abcdefghjklmnopqrstuvwxyz"

def toonehot(text):
    labellist = []
    for letter in text:
        onehot = [0] * len(LETTERSTR)
        num = LETTERSTR.find(letter)
        if num != -1:
            onehot[num] = 1
        labellist.append(onehot)
    return labellist

# Create CNN Model
print("Creating CNN model...")
input = Input((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
out = input
out = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(DROPOUT_RATE)(out)
out = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(DROPOUT_RATE)(out)
out = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Dropout(DROPOUT_RATE)(out)
out = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(out)
out = BatchNormalization()(out)
out = MaxPooling2D(pool_size=(2, 2))(out)
out = Flatten()(out)
out = Dropout(DROPOUT_RATE)(out)
out = [Dense(len(LETTERSTR), name='digit1', activation='softmax')(out),\
    Dense(len(LETTERSTR), name='digit2', activation='softmax')(out),\
    Dense(len(LETTERSTR), name='digit3', activation='softmax')(out),\
    Dense(len(LETTERSTR), name='digit4', activation='softmax')(out)]
model = Model(inputs=input, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] * 4)
model.summary()

print("Reading training data...")
traincsv = open('./data/4_imitate_train_set/captcha_train.csv', 'r', encoding = 'utf8')
train_data = np.stack([np.array(Image.open("./data/4_imitate_train_set/" + row[0] + ".png"))/255.0 for row in csv.reader(traincsv)])
traincsv = open('./data/4_imitate_train_set/captcha_train.csv', 'r', encoding = 'utf8')
read_label = [toonehot(row[1]) for row in csv.reader(traincsv)]
train_label = [[] for _ in range(4)]
for arr in read_label:
    for index in range(4):
        train_label[index].append(arr[index])
train_label = [arr for arr in np.asarray(train_label)]
print("Shape of train data:", train_data.shape)

print("Reading validation data...")
valicsv = open('./data/4_imitate_vali_set/captcha_vali.csv', 'r', encoding = 'utf8')
vali_data = np.stack([np.array(Image.open("./data/4_imitate_vali_set/" + row[0] + ".png"))/255.0 for row in csv.reader(valicsv)])
valicsv = open('./data/4_imitate_vali_set/captcha_vali.csv', 'r', encoding = 'utf8')
read_label = [toonehot(row[1]) for row in csv.reader(valicsv)]
vali_label = [[] for _ in range(4)]
for arr in read_label:
    for index in range(4):
        vali_label[index].append(arr[index])
vali_label = [arr for arr in np.asarray(vali_label)]
print("Shape of validation data:", vali_data.shape)

filepath="./data/model/imitate_4_model.keras"
checkpoint = ModelCheckpoint(filepath, monitor='val_digit4_accuracy', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_digit4_accuracy', patience=5, verbose=1, mode='max')
tensorBoard = TensorBoard(log_dir="./logs", histogram_freq=1)
callbacks_list = [checkpoint, earlystop, tensorBoard]
model.fit(train_data, train_label, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, validation_data=(vali_data, vali_label), callbacks=callbacks_list)
