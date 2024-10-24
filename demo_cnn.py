from keras.models import load_model
from keras.models import Model
from keras import backend as K
from PIL import Image
import numpy as np
import csv
LETTERSTR = "abcdefghjklmnopqrstuvwxyz"


def toonehot(text):
    labellist = []
    for letter in text:
        onehot = [0 for _ in range(34)]
        num = LETTERSTR.find(letter)
        onehot[num] = 1
        labellist.append(onehot)
    return labellist


print("Loading test data...")
testcsv = open('./data/manual_label/captcha_test.csv', 'r', encoding = 'utf8')
test_data = np.stack([np.array(Image.open("./data/manual_label/" + row[0] + ".png"))/255.0 for row in csv.reader(testcsv)])
testcsv = open('./data/manual_label/captcha_test.csv', 'r', encoding = 'utf8')
test_label = [row[1] for row in csv.reader(testcsv)]
print("Loading model...")
K.clear_session()
model = None
model4 = load_model("./data/model/imitate_4_model.keras")
print("Predicting...")
prediction4 = model4.predict(test_data) # 4碼

# 以下計算各個模型各個字元辨識率等等，有點亂，以後有空再整理
total4, total = 0, len(test_label)
correct4, correct = 0, 0
correct4digit = [0 for _ in range(4)]
totalalpha, correctalpha = len([1 for ans in test_label for char in ans if char.isalpha()]), 0
for i in range(total):
    checkcorrect = True
    total4 += 1
    allequal = True
    answer = ""
    for char in range(4):
        answer += LETTERSTR[np.argmax(prediction4[char][i])]
        if LETTERSTR[np.argmax(prediction4[char][i])] == test_label[i][char]:
            correct4digit[char] += 1
            correctalpha += 1 if LETTERSTR[np.argmax(prediction4[char][i])].isalpha() else 0
        else:
            allequal = False
    print("AI解析:" + answer + " 答案:" + test_label[i])
    if allequal:
        correct4 += 1
    else:
        checkcorrect = False
    if checkcorrect:
        correct += 1


print("4digits model acc:{:.4f}%".format(correct4/total4*100)) # 4模型acc
for i in range(4):
    print("digit{:d} acc:{:.4f}%".format(i+1, correct4digit[i]/total4*100)) # 4模型各字元acc
print("---------------------------")
print("alpha acc:{:.4f}%".format(correctalpha/totalalpha*100)) # 整體英文字acc
