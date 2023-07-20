#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=/root/food/build_lmdb
# train.txt和test.txt文件放置的位置
DATA=/root/food/build_lmdb
# caffe/build/tools的位置
TOOLS=/root/caffe/build/tools

# 训练集和测试集的位置，记得，最后的 '/' 不要漏了
TRAIN_DATA_ROOT=/root/food/build_lmdb/train_data/
VAL_DATA_ROOT=/root/food/build_lmdb/test_data/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=224
  RESIZE_WIDTH=224
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

rm -rf $EXAMPLE/ilsvrc12_train_lmdb
rm -rf $EXAMPLE/ilsvrc12_val_lmdb

# $EXAMPLE/ilsvrc12_train_lmdb 中 ilsvrc12_train_lmdb 为LMDB的命名，可以按需更改
# $DATA/train.txt \ 这个txt与自己生成train.txt名字相对应，不然得更改
# $DATA/test.txt \ 同上
# $EXAMPLE/ilsvrc12_val_lmdb 同上

echo "Creating train lmdb..."
GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $EXAMPLE/ilsvrc12_train_lmdb

echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    $VAL_DATA_ROOT \
    $DATA/test.txt \
    $EXAMPLE/ilsvrc12_val_lmdb

echo "Done."
