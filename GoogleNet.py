""" Python version: 3.6.8 | Torch version: 1.1.0 """
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from operator import mul
from functools import reduce
from torchvision import datasets
from torchvision import models

import data_utilities


""" [todo] encapsulate """
mul_ = lambda nums: reduce(mul, nums)


class GoogleNet(nn.Module):
    """
    [MetaInfo]
    1. designed for ImageNet Challenge.
    2. label format is designed to be one-hot.
    3. # of layers: 22
    4. using stride + zero padding to maintain the spatial dimensions.
    """

    """ Note that this implementation is not exactly the original Inception version from
    its paper `Going deeper with convolutions` (InceptionV1) ——
        1. the original version used LRN in InceptionV1, which is not used here. """

    class InceptionModule(nn.Module):
        def __init__(self,
                            in_channels,
                            _1mul1_out_channels,
                            _3mul3_reduce_out_channels,
                            _3mul3_out_channels,
                            _5mul5_reduce_out_channels,
                            _5mul5_out_channels,
                            _pool_proj_out_channels):
            super(GoogleNet.InceptionModule, self).__init__()
            self.conv1_mul_1 = nn.Conv2d(in_channels, _1mul1_out_channels, 1)

            self.conv3_mul_3_reduce = nn.Conv2d(in_channels, _3mul3_reduce_out_channels, 1)
            self.con3_mul_3 = nn.Conv2d(_3mul3_reduce_out_channels,
                                                        _3mul3_out_channels, 3, padding=1)

            self.conv5_mul_5_reduce = nn.Conv2d(in_channels, _5mul5_reduce_out_channels, 1)
            self.conv5_mul_5 = nn.Conv2d(_5mul5_reduce_out_channels,
                                                          _5mul5_out_channels, 5, padding=2)

            self.pool_projection = nn.Conv2d(in_channels, _pool_proj_out_channels, 1)
            return


        """ `forward` signature is required by `__call__`, so you cannot change name. """
        def forward(self, inputs):
            branch1_outputs = F.relu(self.conv1_mul_1(inputs))

            branch2_outputs = F.relu(self.conv3_mul_3_reduce(inputs))
            branch2_outputs = F.relu(self.con3_mul_3(branch2_outputs))

            branch3_outputs = F.relu(self.conv5_mul_5_reduce(inputs))
            branch3_outputs = F.relu(self.conv5_mul_5(branch3_outputs))

            """ note that in pytorch, the padding is done with top-left and bottom-right 
            both every time. """
            branch4_outputs = F.max_pool2d(inputs, (3, 3), stride=1, padding=1)
            branch4_outputs = F.relu(self.pool_projection(branch4_outputs))

            return torch.cat((branch1_outputs,
                                    branch2_outputs,
                                    branch3_outputs,
                                    branch4_outputs), dim=1)


    class AuxClassificationModule(nn.Module):
        __counter = -1

        def __init__(self, google_net, in_channels, out_features):
            super(GoogleNet.AuxClassificationModule, self).__init__()
            GoogleNet.AuxClassificationModule.__counter += 1
            self.conv = nn.Conv2d(in_channels, 128, 1)

            """ because the `in_channels` of `fc1` is required to calculate dynamically 
            as soon as `conv` is defined, I have to know the input dimension of this 
            `AuxClassificationModule`,  which is gotten in real time using the passed 
            `google_net` instance. 
            (alternative: design `Aux..` like a Network, with `feature_shape`, 
            `num_of_classes`, this way `google_net` is replaced with 
            direct outputs shape of `inception_4a` to do data assertion) """
            """ why don't I just calculate using formula? cuz I want both, to do data assertion. """
            """ `2**4` has been tested. """
            _ = lambda dim: (dim // 2 ** 4 - 2) // 3
            fc1_in_features = _(google_net.feature_shape[1]) * \
                                      _(google_net.feature_shape[2]) * \
                                      128
            """ [todo] there is a waste when calculating logits of `outputs_No.0` and `outputs_No.1`. """
            fc1_in_features_2 = google_net.logits(google_net._test_tensor,
                                                                  outputs_No=GoogleNet.AuxClassificationModule.__counter,
                                                                  conv_module_only=True).shape[-1]
            assert fc1_in_features == \
                     fc1_in_features_2, \
                'shape of outputs tensor of `inception_4a(d)` is not correct! check you convolutional params if any. '

            self.fc1 = nn.Linear(fc1_in_features, 1024)
            self.fc2 = nn.Linear(1024, out_features)
            return


        def forward(self, inputs, conv_module_only=False, apply_dropout=True):
            outputs = F.avg_pool2d(inputs, (5, 5), stride=3)
            outputs = F.relu(self.conv(outputs))
            outputs = outputs.view(-1, mul_(outputs.shape[1:]))
            if conv_module_only:
                return outputs
            outputs = F.relu(self.fc1(outputs))
            if apply_dropout:
                outputs = nn.Dropout(p=0.7)(outputs)
            outputs = F.relu(self.fc2(outputs))
            return outputs


    def __init__(self, feature_shape=(3, 224, 224), num_of_classes=1000):
        super(GoogleNet, self).__init__()

        self.feature_shape = feature_shape
        self.num_of_classes = num_of_classes
        self._test_tensor = torch.zeros((1, *feature_shape))
        self.__data_assert__(self._test_tensor)

        """ same padding to be more compatible with images of random size:
        ∵ ⌊X + a⌋ == ⌊X⌋ + a, s.t. isinstance(a, int) == True
        ∴ ⌊(X+P-7)/2⌋ + 2/2 = ⌊X/2⌋ => P == 5. """
        """ [todo] using four-tuple of padding? """
        self.conv1 = nn.Conv2d(feature_shape[0], 64, 7, stride=2, padding=3)
        self.conv2_reduce = nn.Conv2d(64, 64, 1)
        """ in a same way, P == 2. """
        self.conv2 = nn.Conv2d(64, 192, 3, padding=1)

        self.inception_3a = GoogleNet.InceptionModule(in_channels=192,
                                                                             _1mul1_out_channels=64,
                                                                             _3mul3_reduce_out_channels=96,
                                                                             _3mul3_out_channels=128,
                                                                             _5mul5_reduce_out_channels=16,
                                                                             _5mul5_out_channels=32,
                                                                             _pool_proj_out_channels=32)
        self.inception_3b = GoogleNet.InceptionModule(in_channels=256,
                                                                             _1mul1_out_channels=128,
                                                                             _3mul3_reduce_out_channels=128,
                                                                             _3mul3_out_channels=192,
                                                                             _5mul5_reduce_out_channels=32,
                                                                             _5mul5_out_channels=96,
                                                                             _pool_proj_out_channels=64)

        self.inception_4a = GoogleNet.InceptionModule(in_channels=480,
                                                                             _1mul1_out_channels=192,
                                                                             _3mul3_reduce_out_channels=96,
                                                                             _3mul3_out_channels=208,
                                                                             _5mul5_reduce_out_channels=16,
                                                                             _5mul5_out_channels=48,
                                                                             _pool_proj_out_channels=64)
        self._0th_auxiliary_module = GoogleNet.AuxClassificationModule(self, 
                                                                                                        in_channels=512,
                                                                                                        out_features=self.num_of_classes)
        self.inception_4b = GoogleNet.InceptionModule(in_channels=512,
                                                                             _1mul1_out_channels=160,
                                                                             _3mul3_reduce_out_channels=112,
                                                                             _3mul3_out_channels=224,
                                                                             _5mul5_reduce_out_channels=24,
                                                                             _5mul5_out_channels=64,
                                                                             _pool_proj_out_channels=64)
        self.inception_4c = GoogleNet.InceptionModule(in_channels=512,
                                                                             _1mul1_out_channels=128,
                                                                             _3mul3_reduce_out_channels=128,
                                                                             _3mul3_out_channels=256,
                                                                             _5mul5_reduce_out_channels=24,
                                                                             _5mul5_out_channels=64,
                                                                             _pool_proj_out_channels=64)
        self.inception_4d = GoogleNet.InceptionModule(in_channels=512,
                                                                             _1mul1_out_channels=112,
                                                                             _3mul3_reduce_out_channels=144,
                                                                             _3mul3_out_channels=288,
                                                                             _5mul5_reduce_out_channels=32,
                                                                             _5mul5_out_channels=64,
                                                                             _pool_proj_out_channels=64)
        self._1st_auxiliary_module = GoogleNet.AuxClassificationModule(self,
                                                                                                       in_channels=528,
                                                                                                       out_features=self.num_of_classes)
        self.inception_4e = GoogleNet.InceptionModule(in_channels=528,
                                                                             _1mul1_out_channels=256,
                                                                             _3mul3_reduce_out_channels=160,
                                                                             _3mul3_out_channels=320,
                                                                             _5mul5_reduce_out_channels=32,
                                                                             _5mul5_out_channels=128,
                                                                             _pool_proj_out_channels=128)

        self.inception_5a = GoogleNet.InceptionModule(in_channels=832,
                                                                             _1mul1_out_channels=256,
                                                                             _3mul3_reduce_out_channels=160,
                                                                             _3mul3_out_channels=320,
                                                                             _5mul5_reduce_out_channels=32,
                                                                             _5mul5_out_channels=128,
                                                                             _pool_proj_out_channels=128)
        self.inception_5b = GoogleNet.InceptionModule(in_channels=832,
                                                                             _1mul1_out_channels=384,
                                                                             _3mul3_reduce_out_channels=192,
                                                                             _3mul3_out_channels=384,
                                                                             _5mul5_reduce_out_channels=48,
                                                                             _5mul5_out_channels=128,
                                                                             _pool_proj_out_channels=128)
        
        _ = lambda dim: dim // 2 ** 5 - 6
        fc6_in_features = _(self.feature_shape[1]) * \
                                  _(self.feature_shape[2]) * \
                                  1024
        fc6_in_features_2 = self.logits(self._test_tensor,
                                                   conv_module_only=True).shape[-1]
        assert fc6_in_features == \
                 fc6_in_features_2, \
            'shape of outputs tensor of `final pooling outputs` is not correct! ' \
            'check you convolutional params if any. '
        self.fc6 = nn.Linear(fc6_in_features, self.num_of_classes)
        return


    def __data_assert__(self, inputs, labels=None):
        assert isinstance(inputs, torch.Tensor), \
            'inputs must be `torch.Tensor` for maintaining calling standard and ' \
            'design consistency.'
        assert inputs.dim() == 4 and inputs.shape[1:] == self.feature_shape, \
            'inputs of %s model must have shape of ' \
            '(batch_size, %d, %d, %d).' % \
            (self.__class__.__name__, *self.feature_shape)

        if labels is not None:
            assert isinstance(labels, torch.Tensor), \
                'labels must be `torch.Tensor` for maintaining calling standard and' \
                'design consistency.'
            assert labels.dim() == 2 and labels.shape[1:] == (self.num_of_classes, ), \
                'labels of %s model must have shape of ' \
                '(batch_size, %d).' % \
                (self.__class__.__name__, self.num_of_classes)
            assert len(inputs) == len(labels), \
                'sample size is not consistent within `inputs` and `labels`!'
            

    def logits(self, inputs, outputs_No=2, conv_module_only=False, apply_dropout=False):
        """
        :param inputs:
            type: `torch.Tensor`, which are assumed with shape of (batch_size, num_of_channels, x, x).
        :param outputs_No:
            type: `int`, valid value for this param:
            `0` | indicate that using the first auxiliary outputs.
            `1` | indicate that using the second auxiliary outputs.
            `2`(default) | indicate that using the final outputs for whole network.
        :param conv_module_only:
            type: `bool`, return flattened outputs of conv module of network if `True`.
        """
        self.__data_assert__(inputs)
        assert outputs_No in (0, 1, 2)
        assert isinstance(conv_module_only, bool) and isinstance(apply_dropout, bool)

        """ Stem Network Section """
        conv1_outputs = F.relu(self.conv1(inputs))
        """ padding not exact accurate for `pool1~4`. """
        pool1_outputs = F.max_pool2d(conv1_outputs, (3, 3), stride=2, padding=1)
        conv2_reduce_outputs = F.relu(self.conv2_reduce(pool1_outputs))
        conv2_outputs = F.relu(self.conv2(conv2_reduce_outputs))
        pool2_outputs = F.max_pool2d(conv2_outputs, (3, 3), stride=2, padding=1)

        """ Inception Module No.3 Section """
        inception_3a_outputs = self.inception_3a(pool2_outputs)
        inception_3b_outputs = self.inception_3b(inception_3a_outputs)
        pool3_outputs = F.max_pool2d(inception_3b_outputs, (3, 3), stride=2, padding=1)

        """ Inception Module No.4 Section (with Auxiliary Classification Module No.0-1 Section) """
        inception_4a_outputs = self.inception_4a(pool3_outputs)
        if outputs_No == 0:
            _0th_auxiliary_module
            _0th_auxiliary_module
            return self._0th_auxiliary_module(inception_4a_outputs, conv_module_only, apply_dropout)
        inception_4b_outputs = self.inception_4b(inception_4a_outputs)
        inception_4c_outputs = self.inception_4c(inception_4b_outputs)
        inception_4d_outputs = self.inception_4d(inception_4c_outputs)
        if outputs_No == 1:
            return self._1st_auxiliary_module(inception_4d_outputs, conv_module_only, apply_dropout)
        inception_4e_outputs = self.inception_4e(inception_4d_outputs)
        pool4_outputs = F.max_pool2d(inception_4e_outputs, (3, 3), stride=2, padding=1)

        """ Inception Module No.5 Section """
        inception_5a_outputs = self.inception_5a(pool4_outputs)
        inception_5b_outputs = self.inception_5b(inception_5a_outputs)
        
        """ Classifier Section """
        pool5_outputs = F.avg_pool2d(inception_5b_outputs, (7, 7), stride=1)
        pool5_outputs = pool5_outputs.view(-1, pool5_outputs.shape[-1])
        if apply_dropout:
            pool5_outputs = nn.Dropout(p=0.4)(pool5_outputs)
        if conv_module_only:
            return pool5_outputs
        logits = self.fc6(pool5_outputs)
        return logits


    def fit(self, inputs, labels, *,
            lr=0.01,
            batch_size=10,
            momentum=0.9,
            num_of_epoches=10,
            calc_loss=nn.CrossEntropyLoss(),
            apply_dropout=True,
            freq_of_epoch_audit=2):
        self.__data_assert__(inputs, labels)
        sample_size = len(inputs)
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)

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
                """ [todo] redundant computations... """
                _0th_aux_logits_batch = self.logits(inputs_batch, outputs_No=0, apply_dropout=apply_dropout)
                _1st_aux_logits_batch = self.logits(inputs_batch, outputs_No=1, apply_dropout=apply_dropout)
                logits_batch = self.logits(inputs_batch, apply_dropout=apply_dropout)
                """ note the acc_batch matches weights updated in last `batch_No`(last iteration), \ 
                with dropout used, causing some inaccuracies maybe. """
                acc_batch = (
                    torch.sum(torch.argmax(logits_batch, 1) == torch.argmax(labels_batch, 1), dtype=torch.float) /
                    len(labels_batch))
                loss_batch_s.append(loss_batch.item())
                acc_batch_s.append(acc_batch.item())

                _labels_batch = data_utilities.one_hot_encode(labels_batch, reverse=True)
                _0th_aux_loss_batch = calc_loss(_0th_aux_logits_batch, _labels_batch)
                _1st_aux_loss_batch = calc_loss(_1st_aux_logits_batch, _labels_batch)
                loss_batch = calc_loss(logits_batch, _labels_batch)
                loss_batch = 0.3 * _0th_aux_loss_batch + 0.3 * _1st_aux_loss_batch + loss_batch
                loss_batch.backward()
                optimizer.step()

            if epoch_No % freq_of_epoch_audit == 0:
                loss_epoch = {
                    'epoch_No': epoch_No,
                    'loss_batch_s': loss_batch_s,
                    'sum_of_loss_batches': sum(loss_batch_s),
                    'total_loss_epoch_audit': calc_loss(self.logits(inputs),
                                                                     data_utilities.one_hot_encode(labels, reverse=True)).item()
                }
                """ collect correct numbers in `accuracy_batch_s` to simplify this? """
                accuracy_epoch = {
                    'epoch_No': epoch_No,
                    'acc_batch_s': acc_batch_s,
                    'average_acc_batches': (sum(acc_batch_s[:-1]) * batch_size +
                                                      acc_batch_s[-1] * (sample_size % batch_size)) /
                                                      sample_size,
                    'total_accuracy_epoch_audit': self.accuracy(inputs, labels)
                }
                loss_epoch_s.append(loss_epoch)
                accuracy_epoch_s.append(accuracy_epoch)

                print('Epoch: %3d' % epoch_No, '| train loss: %.4f' % loss_epoch['total_loss_epoch_audit'],
                                                                '| train accuracy: %.2f' % accuracy_epoch['total_accuracy_epoch_audit'])
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
            torch.sum(
                torch.argmax(self.logits(inputs), 1) == torch.argmax(labels, 1),
                dtype=torch.float
            ) / len(labels)).item()


    def forward(self, x_s):
        return self.logits(x_s)


if __name__ == '__main__':
    import platform

    if platform.system() == 'Linux':
        DATA_PATH = '/usr/bin/data'
    elif platform.system() == 'Windows':
        DATA_PATH = 'D:\Developers\Codes\Data'
    else:
        print('Data directory not found!')
        exit()

    """ note the two transforms cannot exchange order... """
    data_train = datasets.CIFAR10(
        DATA_PATH, train=True, download=True, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )
    data_test = datasets.CIFAR10(
        DATA_PATH, train=False, download=True, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    )
    x_s_train, y_s_train = data_utilities.features_labels_split([data_train[i] for i in range(100)])
    x_s_test, y_s_test = data_utilities.features_labels_split([data_test[i] for i in range(100)])
    y_s_train = data_utilities.one_hot_encode(y_s_train)
    y_s_test = data_utilities.one_hot_encode(y_s_test)

    net = GoogleNet(feature_shape=x_s_train[0].shape, num_of_classes=10)
    net.fit(x_s_train, y_s_train, batch_size=10, num_of_epoches=100)
    print(net)