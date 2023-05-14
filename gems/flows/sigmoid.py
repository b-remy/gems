# Copyright 2018 The TensorFlow Probability Authors.
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
# ============================================================================
"""Sigmoid bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_probability.python.bijectors import bijector


__all__ = [
    "Sigmoid",
]


class Sigmoid(bijector.Bijector):
  """Bijector which computes `Y = g(X) = 1 / (1 + exp(-X))`."""

  def __init__(self, validate_args=False, name="sigmoid"):
    super(Sigmoid, self).__init__(
        forward_min_event_ndims=0,
        validate_args=validate_args,
        name=name)

  def _forward(self, x, **condition_kwargs):
    return tf.sigmoid(x)

  def _inverse(self, y, **condition_kwargs):
    return tf.math.log(y) - tf.math.log1p(-y)

  # We implicitly rely on _forward_log_det_jacobian rather than explicitly
  # implement _inverse_log_det_jacobian since directly using
  # `-tf.log(y) - tf.log1p(-y)` has lower numerical precision.

  def _forward_log_det_jacobian(self, x, **condition_kwargs):
    return -tf.nn.softplus(-x) - tf.nn.softplus(x)
  

class Scale(bijector.Bijector):

    def __init__(self, scale, validate_args=False, name='scale'):
      self.scale = scale
      super(Scale, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          name=name)

    def _forward(self, x, **condition_kwargs):
      return x * self.scale

    def _inverse(self, y, **condition_kwargs):
      return y / self.scale

    def _inverse_log_det_jacobian(self, y, **condition_kwargs):
      return -self._forward_log_det_jacobian(self._inverse(y))

    def _forward_log_det_jacobian(self, x, **condition_kwargs):
      # Notice that we needn't do any reducing, even when`event_ndims > 0`.
      # The base Bijector class will handle reducing for us; it knows how
      # to do so because we called `super` `__init__` with
      # `forward_min_event_ndims = 0`.
      return x
    
class Shift(bijector.Bijector):

    def __init__(self, shift, validate_args=False, name='shift'):
      self.shift = shift
      super(Shift, self).__init__(
          validate_args=validate_args,
          forward_min_event_ndims=0,
          name=name)

    def _forward(self, x, **condition_kwargs):
      return x + self.shift

    def _inverse(self, y, **condition_kwargs):
      return y - self.shift

    def _inverse_log_det_jacobian(self, y, **condition_kwargs):
      return -self._forward_log_det_jacobian(self._inverse(y))

    def _forward_log_det_jacobian(self, x, **condition_kwargs):
      # Notice that we needn't do any reducing, even when`event_ndims > 0`.
      # The base Bijector class will handle reducing for us; it knows how
      # to do so because we called `super` `__init__` with
      # `forward_min_event_ndims = 0`.
      return x