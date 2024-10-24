from PIL import Image, ImageDraw, ImageFont
from random import randint
import os
import csv
FONTPATH = ["./data/font/SpicyRice-Regular.ttf"]
LETTERSTR = "abcdefghjklmnopqrstuvwxyz"


class captchatext:
    def __init__(self, priority, offset, captchalen):
        self.letter = LETTERSTR[randint(0, len(LETTERSTR) - 1)]
        self.angle = randint(-5, 5)
        self.priority = priority
        self.offset = offset
        self.next_offset = 0
        self.captchalen = captchalen


    def draw(self, image, font):
        # 設定顏色和字體
        color = ( 255, 255, 255, 255)
        font = font

        # 創建透明背景的文字圖層
        text_width, text_height = font.getbbox(self.letter)[2:4]
        text = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
        textdraw = ImageDraw.Draw(text)
        textdraw.text((0, 0), self.letter, font=font, fill=color)

        # 旋轉和調整大小
        text = text.rotate(self.angle, expand=True)
        text = text.resize((text.size[0] // 10, text.size[1] // 10))

        # 計算位置並將文字粘貼到基礎圖像上
        location = (self.offset + 10, 20)
        image.paste(text, location, text)

        self.next_offset = text.size[0] - int(text.size[0]/5)

def generate(GENNUM, SAVEPATH, filename="train"):
    # Ensure the directory exists, if not create it
    os.makedirs(SAVEPATH, exist_ok=True)
    # Open the files
    captchacsv = open(SAVEPATH + "captcha_{:s}.csv".format(filename), 'w', encoding='utf8', newline='')
    lencsv = open(SAVEPATH + "len_{:s}.csv".format(filename), 'w', encoding='utf8', newline='')
    letterlist = []
    lenlist = []
    for index in range(1, GENNUM + 1, 1):
        captchastr = ""
        captchalen = 4
        offset = 0
        captcha = Image.new('RGBA', (120, 100), (2, 108, 223, 0))
        font = ImageFont.truetype(FONTPATH[0], randint(400,600))
        for i in range(captchalen):
            newtext = captchatext(i, offset, captchalen)
            newtext.draw(image=captcha ,font=font)
            offset += newtext.next_offset
            captchastr += str(newtext.letter)
        letterlist.append([str(index).zfill(len(str(GENNUM))), captchastr])
        lenlist.append([str(index).zfill(len(str(GENNUM))), captchalen])

        captcha.convert("RGB").save(SAVEPATH + str(index).zfill(len(str(GENNUM))) + ".png", "PNG")
    writer = csv.writer(captchacsv)
    writer.writerows(letterlist)
    writer = csv.writer(lencsv)
    writer.writerows(lenlist)
    captchacsv.close()
    lencsv.close()


if __name__ == "__main__":
    generate(100, "./data/4_imitate_train_set/", filename="train")
    generate(100, "./data/4_imitate_vali_set/", filename="vali")
    generate(100, "./data/manual_label/", filename="test")

