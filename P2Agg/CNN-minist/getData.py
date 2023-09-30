import gzip
import os

import numpy as np
import torchvision
from torchvision import transforms as transforms




class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None  # 训练集
        self.train_label = None  # 标签
        self.train_data_size = None  # 训练数据的大小
        self.test_data = None  # 测试数据集
        self.test_label = None  # 测试的标签
        self.test_data_size = None  # 测试集数据Size

        self._index_in_train_epoch = 0

        # 如何数据集是mnist
        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID)
        elif self.name == 'cifar10':
            self.load_data(isIID)
        elif self.name == 'GTSRB':
            self.GTSRBDataSetConstruct(isIID)
        else:
            pass



    def mnistDataSetConstruct(self, isIID):

        data_dir = r'.\data\MNIST'

        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

        train_images = extract_images(train_images_path)
        print("-" * 5 + "train_images" + "-" * 5)

        print(train_images.shape)
        print('-' * 22 + "\n")
        train_labels = extract_labels(train_labels_path)
        print("-" * 5 + "train_labels" + "-" * 5)
        print(train_labels.shape)
        print('-' * 22 + "\n")

        test_images = extract_images(test_images_path)
        print("-" * 5 + "test_images" + "-" * 5)
        print(test_images.shape)  # (10000, 28, 28, 1)
        print('-' * 22 + "\n")
        test_labels = extract_labels(test_labels_path)
        print("-" * 5 + "test_labels" + "-" * 5)
        print(test_labels.shape)  # (10000, 10) 10000维
        print(train_labels[1:11])
        print('-' * 22 + "\n")

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]


        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1

        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        print(train_images.shape)
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])



        train_images = train_images.astype(np.float32)

        train_images = np.multiply(train_images, 1.0 / 255.0)

        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:

            order = np.arange(self.train_data_size)

            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        else:
            '''
                # numpy.argmax(array, axis) 用于返回一个numpy数组中最大值的索引值。当一组中同时出现几个最大值时，返回第一个最大值的索引值。
                two_dim_array = np.array([[1, 3, 5], [0, 4, 3]])
                max_index_axis0 = np.argmax(two_dim_array, axis = 0) # 找 纵向 最大值的下标 
                max_index_axis1 = np.argmax(two_dim_array, axis = 1) # 找 横向 最大值的下标
                print(max_index_axis0)
                print(max_index_axis1)
                
                # [0 1 0] 
                # [2 1]
            '''
            labels = np.argmax(train_labels, axis=1)


            order = np.argsort(labels)


            print(order.shape)
            print("标签下标排序2")

            self.train_data = train_images[order]
            self.train_label = train_labels[order]
            print(train_labels[2])

        self.test_data = test_images
        self.test_label = test_labels

    def load_data(self, isIID):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                                 transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform)
        train_data = train_set.data  # (50000, 32, 32, 3)
        train_labels = train_set.targets
        train_labels = np.array(train_labels)  # 将标签转化为
        print(type(train_labels))  # <class 'numpy.ndarray'>
        print(train_labels.shape)  # (50000,)

        test_data = test_set.data  # 测试数据
        test_labels = test_set.targets
        test_labels = np.array(test_labels)
        # print()

        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]


        train_images = train_data.reshape(train_data.shape[0],
                                          train_data.shape[1] * train_data.shape[2] * train_data.shape[3])
        print(train_images.shape)

        test_images = test_data.reshape(test_data.shape[0],
                                        test_data.shape[1] * test_data.shape[2] * test_data.shape[3])

        # ---------------------------归一化处理------------------------------#
        train_images = train_images.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:

            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:

            order = np.argsort(train_labels)
            print("标签下标排序")
            print(train_labels[order[20000:20002]])
            self.train_data = train_images[order]
            self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels

        print('self.test_label', self.test_label)
        print(len(self.train_label))


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):

    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(

                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):

    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):

    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(

                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return dense_to_one_hot(labels)


if __name__ == "__main__":

    mnistDataSet = GetDataSet('cifar10', 0)  # test NON-IID
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])
