import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import os
import logging

img_num = 0


def loginfo(img_num):
    print(img_num)
    logging.info(image_id[img_num])


def print_key(event):
    global img_num

    if event.keycode == 113:
        if img_num - 1 >= 0:
            b_image["image"] = image_list[img_num - 1]
            img_num = img_num - 1
            print(img_num)

    if event.keycode == 114:
        if img_num + 1 < len(image_list):
            b_image["image"] = image_list[img_num + 1]
            img_num = img_num + 1
            print(img_num)
    if event.keycode == 116:
        print(img_num)
        logging.info(image_id[img_num])


if __name__ == '__main__':
    win = tk.Tk()  # 窗口
    win.title('记录图像质量差的样本')  # 标题
    dirpath = '/home/hb/work/unetchoose'
    filename = '/home/hb/work/pywork/ndvi2/UNet_NDVI/result/tile_list_log2.log'
    logging.basicConfig(filename=filename, level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

    n = 164230  # --185394

    start = 21000
    end = 21164

    image_id = list(range(n + start, n + end, 1))
    # images=os.listdir(dirpath)
    image_list = [ImageTk.PhotoImage(Image.open(os.path.join(dirpath, '{}.jpg'.format(i)))) for i in image_id]
    # image_list = [ImageTk.PhotoImage(Image.open(os.path.join(dirpath, images[i]))) for i in image_id]

    b_image = Label(win, image=image_list[img_num])
    b_image.grid(row=0, column=0, columnspan=3)
    b_image.bind("<Key>", print_key)
    b_image.focus_set()
    win.mainloop()
