import numpy as np
import caffe

if __name__ == "__main__":  # main方法的入口
    caffe.set_mode_cpu()  # 在CPU上运行
    # 修改后，选择使用caffe中的resnet18预训练模型
    model_def = './deploy.prototxt'
    # 在Caffe中，需要手动定义ResNet18的前向传递，并在需要时提取中间层特征
    model_weights = './resnet-18.caffemodel'
    model = caffe.Net(model_def, model_weights, caffe.TEST)


    def forward(image):
        # 预处理图像
        transformer = caffe.io.Transformer({'data': model.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        # transformer.set_mean('data', np.load('path/to/mean_file.npy').mean(1).mean(1))
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


    # 3.输入数据的格式调整(总共有四类食材的训练数据) 注意: png比jpg格式多了一个透明度的通道
    features = [[]]
    tt = True
    # 3.1 黄瓜数据的格式调整
    dir_path = './pictures/'  # 定义加载或保存图片数据的路径
    dir_save = './pictures_tensor/'  # 指定存储处理后的图片张量的路径
    for i in range(0, 50):
        path = dir_path + '1.cucumber/' + 'cucumber (' + str(i + 1) + ').png'  # 每张图片完整的绝对路径
        img = caffe.io.load_image(path)
        feature = forward(img)
        feature = feature.reshape(-1)
        if tt:
            features = [feature]
            tt = False
        else:
            features = np.concatenate((features, [feature]), axis=0)
    print('cucumber finished')

    # 3.2 土豆数据的格式调整
    dir_path = './pictures/'  # 定义加载或保存图片数据的路径
    dir_save = './pictures_tensor/'  # 指定存储处理后的图片张量的路径
    for i in range(0, 40):
        path = dir_path + '2.potato/' + 'potato (' + str(i + 1) + ').png'  # 每张图片完整的绝对路径
        img = caffe.io.load_image(path)
        feature = forward(img)
        feature = feature.reshape(-1)
        features = np.concatenate((features, [feature]), axis=0)
    print('potato finished')

    # 3.3 西红柿数据的格式调整
    dir_path = './pictures/'  # 定义加载或保存图片数据的路径
    dir_save = './pictures_tensor/'  # 指定存储处理后的图片张量的路径
    for i in range(0, 40):
        path = dir_path + '3.tomato/' + 'tomato (' + str(i + 1) + ').png'  # 每张图片完整的绝对路径
        img = caffe.io.load_image(path)
        feature = forward(img)
        feature = feature.reshape(-1)
        features = np.concatenate((features, [feature]), axis=0)
    print('tomatoes finished')

    # 3.4 茄子数据的格式调整
    dir_path = './pictures/'  # 定义加载或保存图片数据的路径
    dir_save = './pictures_tensor/'  # 指定存储处理后的图片张量的路径
    for i in range(0, 40):
        path = dir_path + '4.eggplant/' + 'eggplant (' + str(i + 1) + ').png'  # 每张图片完整的绝对路径
        img = caffe.io.load_image(path)
        feature = forward(img)
        feature = feature.reshape(-1)
        features = np.concatenate((features, [feature]), axis=0)
    print('eggplant finished')

    # 3.5 竹笋数据的格式调整(测试集)
    dir_path = './pictures/'  # 定义加载或保存图片数据的路径
    dir_save = './pictures_tensor/'  # 指定存储处理后的图片张量的路径
    for i in range(0, 35):
        path = dir_path + '5.bamboo/' + 'bamboo (' + str(i + 1) + ').png'  # 每张图片完整的绝对路径
        img = caffe.io.load_image(path)
        feature = forward(img)
        feature = feature.reshape(-1)
        features = np.concatenate((features, [feature]), axis=0)
    print('bamboo finished')

    # 将特征数据保存为NumPy数组类型
    # features = np.array(features)
    # print(features)
    # print(features.shape)
    # np.save('./feature2.npy', features)
    # torch.save(feature, 'feature1.npy', _use_new_zipfile_serialization=False)  # 保存得到的输出特征张量(格式是1*512的list),路径这里使用的是绝对路径
    np.savetxt('./feature40.txt', features)
    print('save as feature40.txt')
