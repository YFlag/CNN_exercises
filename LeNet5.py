import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from operator import mul
from functools import reduce
from torchvision import datasets

import data_utilities


""" [todo] encapsulate """
mul_ = lambda nums: reduce(mul, nums)


class LeNet5(nn.Module):
    """
    1. designed for classification.
    2. designed for dealing with image data.
    3. label format is designed to be one-hot.
    """

    """ Note that this implementation is not exactly the original LeNet5 version from
    its paper `Gradient-Based Learning Applied to Document Recognition`, whose
    last layer is `RBF layer` instead of `fully-connected layer + Softmax layer`. """

    def __init__(self, feature_shape=(1, 32, 32), num_of_classes=10):
        """ [todo] config of hyper-parameters design. """
        super(LeNet5, self).__init__()

        self.feature_shape = feature_shape
        self.num_of_classes = num_of_classes
        __test__tensor = torch.zeros((1, *feature_shape))
        self.__data_assert__(__test__tensor)

        """ [todo] should I separate kernel weights from `nn.Conv2d`? """
        self.conv1 = nn.Conv2d(feature_shape[0], 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        """ `1` here denotes batch_size. """
        self.fc1 = nn.Linear(self.logits(__test__tensor, conv_module_only=True).shape[-1],
                                    120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_of_classes)
        return


    def __data_assert__(self, inputs, labels=None):
        """ `labels` are assumed to be one-hot encoded. """
        assert isinstance(inputs, torch.Tensor), \
            'inputs must be `torch.Tensor` for maintaining calling standard and ' \
            'design consistency.'
        """ [todo] tuple here? """
        assert inputs.dim() == 4 and inputs.shape[1:] == self.feature_shape, \
            'inputs of LeNet5 model must have shape of ' \
            '(batch_size, %d, %d, %d).' % tuple(self.feature_shape)

        """ note you can't use `if labels` here... """
        if labels is not None:
            assert isinstance(labels, torch.Tensor), \
                'labels must be `torch.Tensor` for maintaining calling standard and' \
                'design consistency.'
            assert labels.dim() == 2 and labels.shape[1:] == (self.num_of_classes, ), \
                'labels of LeNet5 model must be have shape of ' \
                '(batch_size, %d).' % self.num_of_classes
            assert len(inputs) == len(labels), \
                'sample size is not consistent within `inputs` and `labels`!'


    def logits(self, inputs, conv_module_only=False):
        """
        :param `inputs`:
            type: `torch.Tensor`, which are assumed with shape of (batch_size, num_of_channels, x, x).
        :param `conv_module_only`:
            type: `bool`, return flattened outputs of conv module of network if `True`.
        """
        self.__data_assert__(inputs)
        """ why not put `relu` behind `pool1`? """
        """ [todo] naming so many variables is a waste of memory... """
        """ (x, x) -> (x-4, x-4). e.g. (32, 32) -> (28, 28). """
        conv1_outputs = F.relu(self.conv1(inputs))

        """ (x-4, x-4) -> (x-4, x-4) // 2. e.g. (28, 28) -> (14, 14). """
        pool1_outputs = F.max_pool2d(conv1_outputs, (2, 2))

        """ (x-4, x-4) // 2 -> ((x-4)//2-4, (x-4)//2-2). e.g. (14, 14) -> (10, 10). """
        conv2_outputs = F.relu(self.conv2(pool1_outputs))

        """ ((x-4)//2-4, (x-4)//2-2) -> ((x-4)//2 * two times, (x-4)//2 * two times). e.g. (10, 10) -> (5, 5). """
        pool2_outputs = F.max_pool2d(conv2_outputs, (2, 2))
        pool2_outputs = pool2_outputs.view(-1, mul_(pool2_outputs.shape[1:]))

        if conv_module_only:
            return pool2_outputs

        """ 16 * ((x-4)//2 * two times, (x-4)//2 * two times) -> 120. e.g. 16 * (5, 5) -> 120. """
        fc1_outputs = F.relu(self.fc1(pool2_outputs))

        """ 120 -> 84. """
        fc2_outputs = F.relu(self.fc2(fc1_outputs))

        """ 84 -> `self.num_of_classes`. """
        """ variable name: `outputs`? """
        logits = self.fc3(fc2_outputs)
        return logits


    def fit(self, inputs, labels, *,
            lr=0.01,
            batch_size=10,
            momentum=0.9,
            num_of_epoches=10,
            freq_of_epoch_audit=2,
            calc_loss=nn.CrossEntropyLoss()):
        self.__data_assert__(inputs, labels)
        sample_size = len(inputs)
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

        """ [todo] what level of loss should we collect? """
        loss_epoch_s = []
        accuracy_epoch_s = []
        for epoch_No in range(num_of_epoches):
            loss_batch_s = []
            acc_batch_s = []
            for batch_No in range(math.ceil(sample_size / batch_size)):
                """ `out of index` error would not happen in python slicing of list. """
                inputs_batch = inputs[batch_No * batch_size:
                                                (batch_No + 1) * batch_size]
                labels_batch = labels[batch_No * batch_size:
                                                (batch_No + 1) * batch_size]

                optimizer.zero_grad()
                logits_batch = self.logits(inputs_batch)
                loss_batch = calc_loss(logits_batch,
                                                data_utilities.one_hot_encode(labels_batch, reverse=True))
                loss_batch.backward()
                optimizer.step()
                acc_batch = (
                    torch.sum(torch.argmax(logits_batch, 1) == torch.argmax(labels_batch, 1), dtype=torch.float) /
                    len(labels_batch))

                loss_batch_s.append(loss_batch.item())
                acc_batch_s.append(acc_batch.item())

            loss_epoch = {
                'loss_batch_s': loss_batch_s,
                'sum_of_loss_batches': sum(loss_batch_s),
                'total_loss_epoch_audit': calc_loss(self.logits(inputs),
                                                                 data_utilities.one_hot_encode(labels, reverse=True)).item()
            }
            """ collect correct numbers in `accuracy_batch_s` to simplify this? """
            accuracy_epoch = {
                'acc_batch_s': acc_batch_s,
                'average_acc_batches': (sum(acc_batch_s[:-1]) * batch_size +
                                                  acc_batch_s[-1] * (sample_size % batch_size)) /
                                                  sample_size,
                'total_accuracy_epoch_audit': self.accuracy(inputs, labels)
            }
            loss_epoch_s.append(loss_epoch)
            accuracy_epoch_s.append(accuracy_epoch)

            if epoch_No % freq_of_epoch_audit == 0:
                print('Epoch: %3d' % epoch_No, '| train loss: %.4f' % loss_epoch['total_loss_epoch_audit'],
                                                                '| train accuracy: %.2f' % accuracy_epoch['total_accuracy_epoch_audit'])
                # print('test accuracy: ', accuracy_test)
        return loss_epoch_s, accuracy_epoch_s


    def test(self, inputs, labels):
        return


    def predict(self, inputs):
        self.__data_assert__(inputs)
        logits = self.logits(inputs)
        max_indices = torch.argmax(logits, 1)
        predict = logits.new_zeros(logits.shape, dtype=torch.uint8)
        predict[range(len(logits)), max_indices] = 1
        return predict


    def accuracy(self, inputs, labels):
        self.__data_assert__(inputs, labels)
        return (
            torch.sum(torch.argmax(self.logits(inputs), 1) == torch.argmax(labels, 1), dtype=torch.float) /
            len(labels)).item()


    def forward(self, x_s):
        return self.logits(x_s)


if __name__ == '__main__':
    import platform

    if platform.system() == 'Linux':
        # DATA_PATH = '/home/huangqi/Freund\'s_Shrine/Data'
        DATA_PATH = '/usr/bin/data'
    elif platform.system() == 'Windows':
        DATA_PATH = 'D:\Developers\Codes\Data'
    else:
        print('Data directory not found!')
        exit()

    data_train = datasets.CIFAR10(
        DATA_PATH, train=True, download=True, transform=transforms.ToTensor())
    data_test = datasets.CIFAR10(
        DATA_PATH, train=False, download=True, transform=transforms.ToTensor())
    """ `data_train` is instance of class `CIFAR10`, which does not support slicing???? """
    x_s_train, y_s_train = data_utilities.features_labels_split([data_train[i] for i in range(100)])
    x_s_test, y_s_test = data_utilities.features_labels_split([data_test[i] for i in range(100)])
    y_s_train = data_utilities.one_hot_encode(y_s_train)
    y_s_test = data_utilities.one_hot_encode(y_s_test)

    """ where is softmax?? """
    net = LeNet5(feature_shape=x_s_train[0].shape)
    net.fit(x_s_train, y_s_train, num_of_epoches=100)
    print(net)

