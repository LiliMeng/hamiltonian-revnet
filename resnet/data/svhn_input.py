from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import cPickle as cPickle

import numpy as np
import tensorflow as tf
import sys

import scipy.misc
from scipy import io
import six.moves
from six.moves import urllib
from six.moves.urllib.request import urlretrieve
# Global constants describing the SVHN dataset
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_CLASSES = 10
NUM_CHANNEL = 3
NUM_TRAIN_IMG = 73257
NUM_TEST_IMG = 26032

def get_expected_bytes(filename):
    if filename == "train_32x32.mat":
        byte_size = 182040794
    elif filename == "test_32x32.mat":
        byte_size = 64275384
    elif filename == "extra_32x32.mat":
        byte_size = 1329278602
    else:
        raise Exception("Invalid file name " + filename)
    return byte_size


def maybe_download_and_extract(DATA_URL, dest_directory, extracted_filepath):
    """Download and extract the tarball from Alex's website."""
    # https://github.com/tensorflow/models/blob/dac6755b121f1446ec857cd05c2ff53b2fd26b90/tutorials/image/cifar10/cifar10.py

    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        if statinfo.st_size == get_expected_bytes(filename):
        	print("Found and verified", filename)
    	else:
        	raise Exception("Failed to verify " + filename)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, extracted_filepath)
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


train_SVHN_URL = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
extra_SVHN_URL = "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"
test_SVHN_URL = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"

def read_SVHN(data_folder):

	maybe_download_and_extract(
	    train_SVHN_URL, data_folder, extracted_filepath='train_32x32.mat')
	init_train_path = os.path.join(data_folder, 'train_32x32.mat')

	maybe_download_and_extract(
	    extra_SVHN_URL, data_folder, extracted_filepath='extra_32x32.mat')
	extra_path = os.path.join(data_folder, 'extra_32x32.mat')

	maybe_download_and_extract(
	    test_SVHN_URL, data_folder, extracted_filepath='test_32x32.mat')
	test_path = os.path.join(data_folder, 'test_32x32.mat')

	init_train_data_dict = scipy.io.loadmat(init_train_path)
	init_train_data = init_train_data_dict['X']
	init_train_label = init_train_data_dict['y']
	for i in six.moves.range(len(init_train_label)):
		if init_train_label[i] == 10:
			init_train_label[i] = 0

	extra_data_dict = scipy.io.loadmat(extra_path)
	extra_data = extra_data_dict['X']
	extra_label = extra_data_dict['y']
	for i in six.moves.range(len(extra_label)):
		if extra_label[i] == 10:
			extra_label[i] = 0

	train_data = init_train_data
	train_label = init_train_label

	test_data_dict = scipy.io.loadmat(test_path)
	test_data = test_data_dict['X']
	test_label = test_data_dict['y']
	for i in six.moves.range(len(test_label)):
		if test_label[i] == 10:
			test_label[i] = 0

	train_img = np.reshape(train_data, [NUM_TRAIN_IMG, NUM_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH])
	test_img = np.reshape(test_data, [NUM_TEST_IMG, NUM_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH])
	train_img = np.transpose(train_img, [0, 2, 3, 1])  # change format from [B, C, H, W] to [B, H, W, C] for feeding to Tensorflow
	test_img = np.transpose(test_img, [0, 2, 3, 1])

	mean_img = np.mean(np.concatenate([train_img, test_img]), axis=0)

	SVHN_data = {}
	SVHN_data["train_img"] = train_img - mean_img
	SVHN_data["test_img"] = test_img - test_img
	SVHN_data["train_label"] = train_label[:, 0]
	SVHN_data["test_label"] = test_label[:, 0]

	return SVHN_data
