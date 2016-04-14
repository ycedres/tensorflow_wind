# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

class WindDataSet(object):

  def __init__(self, inputs, outputs, fake_data=False, one_hot=False,
               dtype=tf.float32):

    dtype = tf.as_dtype(dtype).base_dtype

    assert inputs.shape[0] == outputs.shape[0], (
        'inputs.shape: %s outputs.shape: %s' % (inputs.shape,
                                               outputs.shape))
    self._num_examples = inputs.shape[0]

    self._inputs = inputs
    self._outputs = outputs
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def inputs(self):
    return self._inputs

  @property
  def outputs(self):
    return self._outputs

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._inputs = self._inputs[perm]
      self._outputs = self._outputs[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._inputs[start:end], self._outputs[start:end]

def read_wind_data_sets(dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()
  from scipy.io import loadmat
  import math
  datos = loadmat('/home/ycedres/PycharmProjects/doctorado/data/minutal_data_sameInstantOfDay_wStart10_wEnd14_h24.mat')
  entradas = datos['k_10_inputs']
  salidas = datos['k_10_outputs']
  salidas = salidas[0]

  entradas = entradas.transpose()
  #salidas = salidas.transpose()
  import numpy as np
  tmp = ~np.isnan(entradas).any(axis=1)
  entradas = entradas[tmp]
  salidas = salidas[tmp]

  tmp = ~np.isnan(salidas)
  entradas = entradas[tmp]
  salidas = salidas[tmp]

  windowSize = entradas.shape[1]
  nwindows = entradas.shape[0]

  trainSize = math.trunc(nwindows*0.7)
  validationSize = math.trunc(nwindows*0.15)
  testSize = validationSize


  # trainSet_x = entradas[:,0:trainSize-1]
  # validationSet_x=entradas[:,trainSize:trainSize+validationSize]
  # testSet_x = entradas[:,trainSize+validationSize+1:-1]

  trainSet_x = entradas[0:trainSize-1]
  validationSet_x=entradas[trainSize:trainSize+validationSize]
  testSet_x = entradas[trainSize+validationSize+1:-1]



  # trainSet_y = salidas[0:trainSize-1]
  # validationSet_y = salidas[trainSize:trainSize+validationSize]
  # testSet_y = salidas[trainSize+validationSize+1:-1]

  trainSet_y = salidas[0:trainSize-1]
  validationSet_y = salidas[trainSize:trainSize+validationSize]
  testSet_y = salidas[trainSize+validationSize+1:-1]

  train_x = trainSet_x
  train_y = trainSet_y

  valid_x = validationSet_x
  valid_y = validationSet_y

  test_x = testSet_x
  test_y = testSet_y

  data_sets.train = WindDataSet(train_x, train_y, dtype=dtype)
  data_sets.validation = WindDataSet(valid_x, valid_y,
                                 dtype=dtype)
  data_sets.test = WindDataSet(test_x, test_y, dtype=dtype)

  return data_sets