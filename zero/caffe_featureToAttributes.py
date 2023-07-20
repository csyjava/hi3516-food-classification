# caffe_featureToAttributes.py

import math
import numpy as np
import os

import lmdb
os.environ['GLOG_minloglevel'] = '0'
#注意：os.environ['GLOG_minloglevel'] = '3'要写在import caffe之前。因为在导入caffe时caffe会加载GLOG。
import caffe
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # environ是一个字符串所对应环境的映像对象
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd

if __name__ == "__main__":  # main方法的入口
    # features = np.load('./feature.npy',
    #                    allow_pickle=True)  # 加载之前提取的某一个图片特征数据,读取到的是图片的ndarry信息，类型是一个list，大小是【500,1,512】
    # features = np.array(features)[:, 0, :]
    with open('./feature40.txt') as f:
        features = f.readlines()
    features=np.array(features)
    attributes = pd.read_csv('./attributes.txt', sep='\s+',
                             names=['1', '2', '3', '4', '5'])
    attributes = np.array(attributes)
    print('the number of the features:')
    print(features.shape[0])
    print('the number of attributes:')
    print(attributes.shape[1])
    for i in range(attributes.shape[1]):  # 0-4 , 每个周期预测一个属性
        env_data = lmdb.open("./data_lmdb")
        txn_data = env_data.begin(write=True)
        env_label = lmdb.open("./label_lmdb")
        txn_label = env_label.begin(write=True)
        # 设置训练数据
        # 将特征值和属性值输入至lmdb中
        # 特征值
        for j in range(features.shape[0]):
            key = str(j).encode()
            print(j)
            value = features[j].tobytes()
            txn_data.put(key, value)
        # 属性值
        for j in range(features.shape[0]):
            key = str(j).encode()
            value_label = []
            if attributes[j][i] == 1:
                value_label = np.array([attributes[j][i], 0], dtype=np.int32)
            else:
                value_label = np.array([attributes[j][i], 1], dtype=np.int32)
            value_label = value_label.tobytes()
            txn_label.put(key, value_label)
        solver = caffe.SGDSolver('./solver_featureToAttribute('+str(i+1)+').prototxt')
        solver.solve()