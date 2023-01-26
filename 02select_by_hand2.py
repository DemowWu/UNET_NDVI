from tkinter import *
from PIL import ImageTk, Image
import os
# PTL包需要pip安装，这个包才能让python加载jpg等其它各式图片
import logging

root = Tk()
root.title(" '.' ")

dirpath = '/media/hb/Elements/unetchoose'
n = 82110
start = 1000
end = 2000
image_id = list(range(start, end, 1))
image_list = [ImageTk.PhotoImage(Image.open(os.path.join(dirpath, '{}.jpg'.format(i)))) for i in image_id]
b_image = Label(root, image=image_list[0])
b_image.grid(row=0, column=0, columnspan=3)

b_number = Label(root, text="1 of 4", bd=3, relief="sunken", anchor=E)
b_number.grid(row=2, column=0, columnspan=3, sticky="we")
filename = '/home/hb/work/xuezhe/ndvi/UNET-ZOO-master/result/tile_list_log.log'
logging.basicConfig(filename=filename, level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')


def forward(img_num):
    global b_image
    global b_forward
    global b_back
    global b_log

    b_image["image"] = image_list[img_num - 1]
    b_forward["command"] = lambda: forward(img_num + 1)
    b_back["command"] = lambda: back(img_num - 1)

    b_log["command"] = lambda: loginfo(img_num - 1)

    # 用lambda能让按钮属性改变，并能传递变量
    if img_num == len(image_list):
        b_forward["state"] = "disabled"
    elif img_num == 1:
        b_back["state"] = "disabled"
    else:
        b_back["state"] = "normal"
        b_forward["state"] = "normal"

    b_number["text"] = str(img_num) + " of " + str(len(image_list))


def back(img_num):
    global b_image
    global b_forward
    global b_back
    global b_log

    b_image["image"] = image_list[img_num - 1]
    b_forward["command"] = lambda: forward(img_num + 1)
    b_back["command"] = lambda: back(img_num - 1)

    b_log["command"] = lambda: loginfo(img_num - 1)

    if img_num == len(image_list):
        b_forward["state"] = "disabled"
    elif img_num == 1:
        b_back["state"] = "disabled"
    else:
        b_back["state"] = "normal"
        b_forward["state"] = "normal"

    b_number["text"] = str(img_num) + " of " + str(len(image_list))


def loginfo(img_num):
    print(img_num)
    logging.info(image_id[img_num])


b_back = Button(root, text="<<", command=back)
b_forward = Button(root, text=">>", command=lambda: forward(2))
b_log = Button(root, text="Log", command=loginfo)

b_back.grid(column=0, row=1)
b_forward.grid(column=2, row=1)
b_log.grid(column=1, row=1)

root.mainloop()
