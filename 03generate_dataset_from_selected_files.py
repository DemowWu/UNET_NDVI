import random
import pickle as pkl
import os


# exclude the cloud list to get the clean data
def get_selected_files():
    n = len(os.listdir('/home/hb/work/unetcut')) // 2
    print(n)
    # n = 60000
    logfile = '/home/hb/work/pywork/ndvi2/UNET-ZOO/data/unet_cloud_data_list2000.txt'
    with open(logfile, 'r') as f1:
        list1 = f1.readlines()
        tiles = []
        for l in list1:
            if 'INFO:' in l:
                ll = l.split('INFO:')
                tiles.append(int(ll[1].strip()))

    kk = set(tiles)
    index_list = []
    for i in range(n):
        if i not in kk:
            index_list.append(i)

    return index_list


# get data from two selected dataset directly
def get_selected_files2():
    dirs = ['/media/hb/data/data/unetselect2']
    index_list = []

    for dirr in dirs:
        d = os.listdir(dirr)
        for i in d:
            if 'mask' not in i:
                index_list.append(int(i.split('.')[0].strip()))

    kk = set(index_list)
    print(len(index_list), len(kk))

    return index_list


def train_val_test_split():
    # n = len(os.listdir(args.data_dir)) // 2  # 因为数据集中一套训练数据包含有训练图和mask图，所以要除2
    # get files selected by hand
    # index_list = get_selected_files()
    index_list = get_selected_files2()
    n = len(index_list)
    random.shuffle(index_list)
    random.shuffle(index_list)
    trains = index_list[0:int(n * 0.7)]
    vals = index_list[int(n * 0.7):int(n * 0.95)]
    tests = index_list[int(n * 0.95):]

    return trains, vals, tests


if __name__ == '__main__':
    print('将train val tests输出为单独的几个文件，方便后续调用时保持一致性')

    a = train_val_test_split()

    pkl.dump(a, open('data/unet_trainset5.pkl', 'wb'))
    tt = pkl.load(open('data/unet_trainset5.pkl', 'rb'))
    print(len(tt[0]), len(tt[1]), len(tt[2]))
    # print(len(tt))
