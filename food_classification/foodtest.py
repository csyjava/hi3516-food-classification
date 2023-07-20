import numpy as np
import math
import os
import lmdb

os.environ['GLOG_minloglevel'] = '3'
# 注意：os.environ['GLOG_minloglevel'] = '3'要写在import caffe之前。因为在导入caffe时caffe会加载GLOG。
import caffe

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # environ是一个字符串所对应环境的映像对象
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd


def forward(image):
    # 预处理图像
    transformer = caffe.io.Transformer({'data': model.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))
    image = transformer.preprocess('data', image)

    # 将图像数据传入网络中进行推理
    model.blobs['data'].data[...] = image
    model.forward()
    # 提取中间层特征
    feature = model.blobs['pool5'].data
    # 返回特征
    return feature


def featureToAttributes(features):
    model_def = './deploy_featureToAttribute.prototxt'
    model = caffe.Net(model_def, model_weights, caffe.TEST)
    trans_to_attribute = caffe.io.Transformer({'data': model.blobs['data'].data.shape})
    features = trans_to_attribute.preprocess('data', features)
    # 将图像数据传入网络中进行推理
    model.blobs['data'].data[...] = features
    model.forward()
    attribute = model.blobs['prob_eval'].data
    # print(attribute)
    if attribute[0][0] > attribute[0][1]:
        attribute = [1]
    else:
        attribute = [0]
    return attribute


def attributesToClasses(attributes):
    model_def = './deploy_attributesToClasses.prototxt'
    model = caffe.Net(model_def, model_weights, caffe.TEST)
    trans_to_classes = caffe.io.Transformer({'data': model.blobs['data'].data.shape})
    attributes=trans_to_classes.preprocess('data', attributes)
    model.blobs['data'].data[...] = attributes
    model.forward()
    class_test = model.blobs['prob_eval'].data
    print('class_test:',class_test)
    return class_test


if __name__ == "__main__":  # main方法的入口
    caffe.set_mode_cpu()  # 在CPU上运行
    # 修改后，选择使用caffe中的resnet18预训练模型
    model_def = './deploy.prototxt'
    # 在Caffe中，需要手动定义ResNet18的前向传递，并在需要时提取中间层特征
    model_weights = './resnet-18.caffemodel'
    model = caffe.Net(model_def, model_weights, caffe.TEST)

    dir_path = './test_picture.png'  # 定义加载或保存图片数据的路径
    dir_save = './pictures_tensor/'  # 指定存储处理后的图片张量的路径
    path = dir_path
    img = caffe.io.load_image(path)
    # 得到特征
    feature = forward(img)
    # 开始映射至属性
    features = np.array(feature)[:, 0, :]

    '''
    初始化属性和类别设置
    '''
    print('the number of the features:')
    print(features.shape[0])
    print('the number of attributes_init: 5')
    print('the number of classes: 5')

    '''
    预测属性
    '''
    attributes = []
    for i in range(5):  # 0-4 , 每个周期预测一个属性
        model_weights = './solver_featureToAttribute(' + str(i + 1) + ')_iter_1.caffemodel'
        attribute = featureToAttributes(features)
        attributes.append(attribute)
    print(attributes)
    '''
    预测类别
    '''
    # 对于每种类别都训练一次，二值判断法
    class_tmp=[]
    model_weights = './solver_attributesToClasses_iter_1.caffemodel'
    attributes = np.array(attributes)
    attributes = attributes.astype(np.float32)
    class_test = attributesToClasses(attributes)
    class_get=class_test[0][0]
    print(class_get)

    print('finsh')
