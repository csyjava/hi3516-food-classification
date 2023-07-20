import os

# 数据集路径
dataset_path = '/root/food/build_lmdb'

# 训练集和测试集文件夹名称
train_folder = 'train_data'
test_folder = 'test_data'

# 标签和文件路径的映射关系
label_map = {}

# 遍历数据集路径下的子文件夹，生成标签和文件路径的映射关系
for folder in [train_folder, test_folder]:
    folder_path = os.path.join(dataset_path, folder)
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue
        for file_name in os.listdir(label_path):
            if not file_name.endswith('.png'):
                continue
            file_path = os.path.join(label, file_name)
            # print(file_path)
            # 将标签和文件路径添加到映射关系中
            label_map[file_path] = label

# 将映射关系写入训练集和测试集的文本文件中
with open('train.txt', 'w') as f:
    for file_path, label in label_map.items():
        if 'train' in file_path:
            f.write('{} {}\n'.format(file_path, label))

with open('test.txt', 'w') as f:
    for file_path, label in label_map.items():
        if 'test' in file_path:
            f.write('{} {}\n'.format(file_path, label))