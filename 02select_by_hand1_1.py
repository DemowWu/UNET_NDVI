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

    if event.keycode == 116:
        if img_num + 1 < len(image_list):
            b_image["image"] = image_list[img_num + 1]
            img_num = img_num + 1
            print(img_num)
    if event.keycode == 114:
        print(img_num)
        logging.info(image_id[img_num])


if __name__ == '__main__':
    win = tk.Tk()  # 窗口
    win.title('记录图像质量差的样本')  # 标题
    dirpath = '/media/hb/Elements/unetchoose'
    filename = '/home/hb/work/pywork/ndvi2/UNET-ZOO/result/tile_selected2.log'
    logging.basicConfig(filename=filename, level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')
    logfile = '/home/hb/work/pywork/ndvi2/UNET-ZOO/data/selected_first_by_remove_cloud_in_mask.log'
    with open(logfile, 'r') as f1:
        list1 = f1.readlines()
        ids=[]
        for l in list1:
            lid=l.split('INFO:')
            ids.append(int(lid[1].strip()))

        list1 = set(ids)
        n = 82110
        start = 78000 #78000
        end = 82110
        kk = 0
        image_id = []
        for i in range(start, end, 1):
            if i not in list1:
                image_id.append(i)

            if len(image_id)>=3000:
                break

            print(i)
        print('list len',len(image_id))
        # image_id = list(range(start, end, 1))
        image_list = [ImageTk.PhotoImage(Image.open(os.path.join(dirpath, '{}.jpg'.format(i)))) for i in image_id]
        b_image = Label(win, image=image_list[img_num])
        b_image.grid(row=0, column=0, columnspan=3)
        b_image.bind("<Key>", print_key)
        b_image.focus_set()
        win.mainloop()
