import math
import numpy as np
import os
import lmdb

os.environ['GLOG_minloglevel'] = '0'
# 注意：os.environ['GLOG_minloglevel'] = '3'要写在import caffe之前。因为在导入caffe时caffe会加载GLOG。
import caffe
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # environ是一个字符串所对应环境的映像对象
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd

if __name__ == "__main__":  # main方法的入口
    attributes = pd.read_csv('./attributes.txt', sep='\s+',
                             names=['1', '2', '3', '4', '5'])
    attributes = np.array(attributes)
    attributes = attributes.astype(float)
    classes = pd.read_csv('./class.txt', sep='\s+', names=['classes'])
    classes = np.array(classes)
    classes = classes.astype(int)
    class_num = np.unique(classes)
    # class_num.size为种类个数
    # classes.shape[0]为总共设下的训练数据数
    # 对于每种类别
    print('the number of attributes')
    print(attributes.shape[0])
    print('the number of classes:')
    print(class_num.size)


    env_attributes = lmdb.open("./attributes_lmdb")
    txn_attributes = env_attributes.begin(write=True)
    env_classes = lmdb.open("./classes_lmdb")
    txn_classes = env_classes.begin(write=True)
# 设置训练数据
# 将属性值和类别值输入至lmdb中
# 属性值
    for j in range(attributes.shape[0]):
        key = str(j).encode()
        # print(attributes[j])
        value = attributes[j].tobytes()
        txn_attributes.put(key, value)
# 类别值
    for j in range(class_num.size):
        key = str(j).encode()
        value_classes = classes[j].tobytes()
        txn_classes.put(key, value_classes)
    solver = caffe.SGDSolver('./solver_attributesToClasses.prototxt')
    solver.solve()

    print('finsh')
