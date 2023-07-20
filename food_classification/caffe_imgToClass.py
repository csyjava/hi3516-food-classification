import numpy as np
import lmdb
from PIL import Image
import cv2
import os
import pickle
os.environ['GLOG_minloglevel'] = '0'
# 注意：os.environ['GLOG_minloglevel'] = '3'要写在import caffe之前。因为在导入caffe时caffe会加载GLOG。
import caffe

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # environ是一个字符串所对应环境的映像对象
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd

if __name__ == "__main__":  # main方法的入口
    solver = caffe.SGDSolver('solver_imgToClass.prototxt')
    solver.solve()
    # env_image = lmdb.open("./image_lmdb")
    # txn_image = env_image.begin(write=True)
    # env_class = lmdb.open("./label_lmdb")
    # txn_class = env_class.begin(write=True)
    # cache = {}

    # 创建数据层
    def create_data_layer(lmdb_path, batch_size):
        data, label = caffe.layers.Data(
            source=lmdb_path,
            backend=caffe.params.Data.LMDB,
            batch_size=batch_size,
            transform_param=dict(
                crop_size=224,
                mean_value=[104, 117, 123],
                mirror=True
            )
        )
        return data, label

    #
    # # 定义网络结构
    # def define_network():
    #     net = caffe.NetSpec()
    #     net.data, net.label = create_data_layer('train_img', 32)
    #     net.conv1 = caffe.layers.Convolution(...)
    #     net.pool1 = caffe.layers.Pooling(...)
    #     ...
    #     net.loss = caffe.layers.SoftmaxWithLoss(...)
    #     return net.to_proto()
    #


    def write_lmdb(img_lmdb_name,data_lamdb_name,data_list):
        # mean = [104, 117, 123]  # RGB 图像的均值值
        env_img = lmdb.open(img_lmdb_name)
        txn_img=env_img.begin(write=True)
        env_data = lmdb.open(data_lamdb_name)
        txn_data = env_data.begin(write=True)
        for i, (image_path, label) in enumerate(data_list):
            with open(image_path, 'rb') as f:
                    img = Image.open(f)
                    img = img.convert('RGB')
                    img = img.resize((224, 224))
                    img = np.array(img)
                    img = img.transpose((2, 0, 1))
                    print(img.shape)
                    img=pickle.dumps(img)
                    txn_img.put(str(i).encode('ascii'),img)
                    txn_data.put(str(label).encode('ascii'))
                #     datum = caffe.proto.caffe_pb2.Datum()
                #     datum.channels = img.shape[0]
                #     datum.height = img.shape[1]
                #     datum.width = img.shape[2]
                #     datum.data = img.tobytes()
                #     datum.label = int(label)
                #     str_id = '{:08}'.format(i).encode('ascii')
                # txn.put(str_id, datum.SerializeToString())

    #
    # # 3.1 黄瓜数据的格式调整
    # dir_path = './pictures/'  # 定义加载或保存图片数据的路径
    # dir_save = './pictures_tensor/'  # 指定存储处理后的图片张量的路径
    # data_list = []
    # for i in range(0,100):
    #     path = dir_path + '1.cucumber/' + 'cucumber (' + str(i + 1) + ').png'  # 每张图片完整的绝对路径
    #     img = caffe.io.load_image(path)
    #     data_list.append((path,1))
    #     # label_list.append((1))
    # print('cucumber finished')
    # # 3.2 土豆数据的格式调整
    # dir_path = './pictures/'  # 定义加载或保存图片数据的路径
    # dir_save = './pictures_tensor/'  # 指定存储处理后的图片张量的路径
    # for i in range(0,100):
    #     path = dir_path + '2.potato/' + 'potato (' + str(i + 1) + ').png'  # 每张图片完整的绝对路径
    #     img = caffe.io.load_image(path)
    #     data_list.append((path, 2))
    # print('potato finished')
    # # 3.3 西红柿数据的格式调整
    # dir_path = './pictures/'  # 定义加载或保存图片数据的路径
    # dir_save = './pictures_tensor/'  # 指定存储处理后的图片张量的路径
    # for i in range(0,100):
    #     path = dir_path + '3.tomato/' + 'tomato (' + str(i + 1) + ').png'  # 每张图片完整的绝对路径
    #     img = caffe.io.load_image(path)
    #     data_list.append((path, 3))
    # print('tomatoes finished')
    #
    # # 3.4 茄子数据的格式调整
    # dir_path = './pictures/'  # 定义加载或保存图片数据的路径
    # dir_save = './pictures_tensor/'  # 指定存储处理后的图片张量的路径
    # for i in range(0,100):
    #     path = dir_path + '4.eggplant/' + 'eggplant (' + str(i + 1) + ').png'  # 每张图片完整的绝对路径
    #     img = caffe.io.load_image(path)
    #     data_list.append((path, 4))
    # print('eggplant finished')
    #
    # # 3.5 竹笋数据的格式调整(测试集)
    # dir_path = './pictures/'  # 定义加载或保存图片数据的路径
    # dir_save = './pictures_tensor/'  # 指定存储处理后的图片张量的路径
    # for i in range(0, 100):
    #     path = dir_path + '5.bamboo/' + 'bamboo (' + str(i + 1) + ').png'  # 每张图片完整的绝对路径
    #     img = caffe.io.load_image(path)
    #     data_list.append((path, 5))
    # print('bamboo finished')
    #
    # write_lmdb('train_img','train_label', data_list)

    # 训练网络
    #
    # def forward(image):
    #     # 预处理图像
    #     transformer = caffe.io.Transformer({'data': model.blobs['data'].data.shape})
    #     transformer.set_transpose('data', (2, 0, 1))
    #     # transformer.set_mean('data', np.load('path/to/mean_file.npy').mean(1).mean(1))
    #     transformer.set_raw_scale('data', 255)
    #     transformer.set_channel_swap('data', (2, 1, 0))
    #     image = transformer.preprocess('data', image)
    #
    #     # 将图像数据传入网络中进行推理
    #     model.blobs['data'].data[...] = image
    #     model.forward()
    #     # 提取中间层特征
    #     result = model.blobs['prob'].data
    #     # 返回特征
    #     return result
