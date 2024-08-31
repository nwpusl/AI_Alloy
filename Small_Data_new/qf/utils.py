import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
# import graphviz
import os
import warnings

warnings.filterwarnings('ignore')

# 绘图字体设置
plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'Times New Roman']  # 汉字字体集
plt.rcParams['font.size'] = 12  # 字体大小
plt.rcParams['axes.unicode_minus'] = False

os.environ['PATH'] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin\\'
test_ratio = 0.3

def create_excel(path):

    if 'xlsx' in path:
        pd.DataFrame().to_excel(path)
    else:
        print('表格路径非法，请重新检查')
# 　文件内容的复制函数
def mycopyfile(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        # print(dstpath+fname)
        shutil.copy(srcfile, dstpath + '/' + fname)  # 复制文件
        # print ("copy %s -> %s"%(srcfile, dstpath + fname))


# 　文件夹的复制函数
def mycopydir(srcdir, dstdir):
    for dirpath, dirnames, filenames in os.walk(srcdir):
        # print('dirpath=',dirpath)
        # print('dirnames=',dirnames)
        # print('file_names=',filenames)
        for file in filenames:
            path_file = os.path.join(dirpath, file)
            mycopyfile(path_file, dstdir)


# 创建新的文件夹，分为主文件夹和子文件夹
def create_dir(save_path, is_mainpath=False):
    # 对于main_path如果不存在就创建
    if is_mainpath:
        if os.path.exists(save_path):
            print(f'文件夹{save_path}已存在')
        else:
            print(f'创建{save_path}文件夹')
            os.mkdir(save_path)
    # 对于非main_path则对每一个main_path创建一个备份文件夹
    else:
        if os.path.exists(save_path):
            print(f'文件夹{save_path}已存在')
        else:
            print(f'创建{save_path}文件夹')
            os.mkdir(save_path)

        if os.path.exists(save_path + '(temp)'):
            print(f'文件夹{save_path}已存在')
        else:
            print(f'创建{save_path}文件夹')
            os.mkdir(save_path + '(temp)')


def create_empty_ls(num):
    if not isinstance(num, float) and not isinstance(num, int):
        print('数组长度不是整数')
        return
    else:
        ls = list(np.zeros(num))
        for i in range(num):
            ls[i] = []
        return ls
