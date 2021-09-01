# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

"""Wide ResNet with HetSNGP."""
import functools
from absl import logging

import edward2 as ed
import tensorflow as tf
import tensorflow_probability as tfp


class MCSoftmaxSNGP(ed.layers.MCSoftmaxDense):
  """MC estimation of softmax approx to heteroscedastic SNGP predictions."""

  def __init__(self,
               num_classes,
               logit_noise=tfp.distributions.Normal,
               temperature=1.0,
               train_mc_samples=1000,
               test_mc_samples=1000,
               compute_pred_variance=False,
               share_samples_across_batch=False,
               logits_only=False,
               eps=1e-7,
               dtype=None,
               kernel_regularizer=None,
               bias_regularizer=None,
               num_inducing=1024,
               gp_kernel_type='gaussian',
               gp_kernel_scale=1.,
               gp_output_bias=0.,
               normalize_input=False,
               gp_kernel_scale_trainable=False,
               gp_output_bias_trainable=False,
               gp_cov_momentum=-1.,
               gp_cov_ridge_penalty=1.,
               scale_random_features=True,
               use_custom_random_features=True,
               custom_random_features_initializer=None,
               custom_random_features_activation=None,
               l2_regularization=1e-6,
               gp_cov_likelihood='gaussian',
               return_gp_cov=True,
               return_random_features=False,
               sngp_var_weight=0.,
               het_var_weight=1.,
               name='MCSoftmaxSNGP'):
    """Initializes an MCSoftmaxSNGP layer instance.

    Args:
      num_classes: Integer. Number of classes for classification task.
      logit_noise: tfp.distributions instance. Must be a location-scale
        distribution. Valid values: tfp.distributions.Normal,
        tfp.distributions.Logistic, tfp.distributions.Gumbel.
      temperature: Float or scalar `Tensor` representing the softmax
        temperature.
      train_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during training.
      test_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution during testing/inference.
      compute_pred_variance: Boolean. Whether to estimate the predictive
        variance. If False the __call__ method will output None for the
        predictive_variance tensor.
      share_samples_across_batch: Boolean. If True, the latent noise samples
        are shared across batch elements. If encountering XLA compilation errors
        due to dynamic shape inference setting = True may solve.
      logits_only: Boolean. If True, only return the logits from the __call__
        method. Set True to serialize tf.keras.Sequential models.
      eps: Float. Clip probabilities into [eps, 1.0] before applying log.
      dtype: Tensorflow dtype. The dtype of output Tensor and weights associated
        with the layer.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      num_inducing: (int) Number of random Fourier features used for
        approximating the Gaussian process.
      gp_kernel_type: (string) The type of kernel function to use for Gaussian
        process. Currently default to 'gaussian' which is the Gaussian RBF
        kernel.
      gp_kernel_scale: (float) The length-scale parameter of the a
        shift-invariant kernel function, i.e., for RBF kernel:
        exp(-|x1 - x2|**2 / gp_kernel_scale).
      gp_output_bias: (float) Scalar initial value for the bias vector.
      normalize_input: (bool) Whether to normalize the input to Gaussian
        process.
      gp_kernel_scale_trainable: (bool) Whether the length scale variable is
        trainable.
      gp_output_bias_trainable: (bool) Whether the bias is trainable.
      gp_cov_momentum: (float) A discount factor used to compute the moving
        average for posterior covariance matrix.
      gp_cov_ridge_penalty: (float) Initial Ridge penalty to posterior
        covariance matrix.
      scale_random_features: (bool) Whether to scale the random feature
        by sqrt(2. / num_inducing).
      use_custom_random_features: (bool) Whether to use custom random
        features implemented using tf.keras.layers.Dense.
      custom_random_features_initializer: (str or callable) Initializer for
        the random features. Default to random normal which approximates a RBF
        kernel function if activation function is cos.
      custom_random_features_activation: (callable) Activation function for the
        random feature layer. Default to cosine which approximates a RBF
        kernel function.
      l2_regularization: (float) The strength of l2 regularization on the output
        weights.
      gp_cov_likelihood: (string) Likelihood to use for computing Laplace
        approximation for covariance matrix. Default to `gaussian`.
      return_gp_cov: (bool) Whether to also return GP covariance matrix.
        If False then no covariance learning is performed.
      return_random_features: (bool) Whether to also return random features.
      sngp_var_weight: (float) Mixing weight for the SNGP variance in the
        total variances during testing.
      het_var_weight: (float) Mixing weight for the heteroscedastic variance
        in the total variance during testing.
      name: (str) The name of the layer used for name scoping.
    """
    super().__init__(num_classes=num_classes,
                     logit_noise=logit_noise,
                     temperature=temperature,
                     train_mc_samples=train_mc_samples,
                     test_mc_samples=test_mc_samples,
                     compute_pred_variance=compute_pred_variance,
                     share_samples_across_batch=share_samples_across_batch,
                     logits_only=logits_only,
                     eps=eps,
                     dtype=dtype,
                     kernel_regularizer=kernel_regularizer,
                     bias_regularizer=bias_regularizer,
                     name=name)
    self.sngp_layer = ed.layers.RandomFeatureGaussianProcess(
        units=num_classes,
        num_inducing=num_inducing,
        gp_kernel_type=gp_kernel_type,
        gp_kernel_scale=gp_kernel_scale,
        gp_output_bias=gp_output_bias,
        normalize_input=normalize_input,
        gp_kernel_scale_trainable=gp_kernel_scale_trainable,
        gp_output_bias_trainable=gp_output_bias_trainable,
        gp_cov_momentum=gp_cov_momentum,
        gp_cov_ridge_penalty=gp_cov_ridge_penalty,
        scale_random_features=scale_random_features,
        use_custom_random_features=use_custom_random_features,
        custom_random_features_initializer=custom_random_features_initializer,
        custom_random_features_activation=custom_random_features_activation,
        l2_regularization=l2_regularization,
        gp_cov_likelihood=gp_cov_likelihood,
        return_gp_cov=return_gp_cov,
        return_random_features=return_random_features,
        dtype=dtype,
        name='SNGP_layer')
    self.sngp_var_weight = sngp_var_weight
    self.het_var_weight = het_var_weight

  def _compute_loc_param(self, inputs, training):
    """Computes the mean logits as the mean-field logits of the SNGP."""
    logits_sngp, covmat_sngp = self.sngp_layer(inputs)
    # logits_mean = ed.layers.utils.mean_field_logits(
    #     logits=logits_sngp,
    #     covmat=covmat_sngp,
    #     mean_field_factor=self.gp_mean_field_factor)
    return logits_sngp, covmat_sngp

  def _compute_scale_param(self, inputs, covmat_sngp, training):
    """Computes the variances for the logits."""
    het_scales = super()._compute_scale_param(inputs)
    sngp_marginal_vars = tf.expand_dims(tf.linalg.diag_part(covmat_sngp), -1)
    # sngp_marginal_vars.shape = (batch_size, 1),
    # het_scales.shape = (batch_size, num_classes)
    if training:
      joint_scales = het_scales
    else:
      joint_scales = tf.sqrt(self.het_var_weight * tf.square(het_scales)
                             + self.sngp_var_weight * sngp_marginal_vars)
      # new version, as discussed in our meeting (to be checked for correctness)
      # sngp_prec = (1. / sngp_marginal_vars) - 1.
      # joint_scales = tf.sqrt(1. / (sngp_prec + tf.square(het_scales)))
    return joint_scales

  def __call__(self, inputs, training=True, seed=None):
    """Computes predictive and log predictive distribution.

    Uses Monte Carlo estimate of softmax approximation to heteroscedastic model
    to compute predictive distribution. O(mc_samples * num_classes).

    Args:
      inputs: Tensor. The input to the heteroscedastic output layer.
      training: Boolean. Whether we are training or not.
      seed: Python integer for seeding the random number generator.

    Returns:
      Tensor logits if logits_only = True. Otherwise,
      tuple of (logits, log_probs, probs, predictive_variance). For multi-class
      classification i.e. num_classes > 2 logits = log_probs and logits can be
      used with the standard tf.nn.sparse_softmax_cross_entropy_with_logits loss
      function. For binary classification i.e. num_classes = 2, logits
      represents the argument to a sigmoid function that would yield probs
      (logits = inverse_sigmoid(probs)), so logits can be used with the
      tf.nn.sigmoid_cross_entropy_with_logits loss function.

    Raises:
      ValueError if seed is provided but model is running in graph mode.
    """
    # Seed shouldn't be provided in graph mode.
    if not tf.executing_eagerly():
      if seed is not None:
        raise ValueError('Seed should not be provided when running in graph '
                         'mode, but %s was provided.' % seed)
    with tf.name_scope(self._name):
      locs, covmat = self._compute_loc_param(inputs, training)  # pylint: disable=assignment-from-none
      scale = self._compute_scale_param(inputs, covmat, training)  # pylint: disable=assignment-from-none

      if training:
        total_mc_samples = self._train_mc_samples
      else:
        total_mc_samples = self._test_mc_samples

      probs_mean = self._compute_predictive_mean(
          locs, scale, total_mc_samples, seed)

      pred_variance = None
      if self._compute_pred_variance:
        pred_variance = self._compute_predictive_variance(
            probs_mean, locs, scale, seed, total_mc_samples)

      probs_mean = tf.clip_by_value(probs_mean, self._eps, 1.0)
      log_probs = tf.math.log(probs_mean)

      if self._num_classes == 2:
        # inverse sigmoid
        probs_mean = tf.clip_by_value(probs_mean, self._eps, 1.0 - self._eps)
        logits = log_probs - tf.math.log(1.0 - probs_mean)
      else:
        logits = log_probs

      if self._logits_only:
        return logits

      return logits, log_probs, probs_mean, pred_variance

  def reset_covariance_matrix(self):
    """Resets the covariance matrix of the SNGP layer."""
    self.sngp_layer.reset_covariance_matrix()


# pylint: disable=invalid-name
BatchNormalization = functools.partial(
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)


def make_random_feature_initializer(random_feature_type):
  # Use stddev=0.05 to replicate the default behavior of
  # tf.keras.initializer.RandomNormal.
  if random_feature_type == 'orf':
    return ed.initializers.OrthogonalRandomFeatures(stddev=0.05)
  elif random_feature_type == 'rff':
    return tf.keras.initializers.RandomNormal(stddev=0.05)
  else:
    return random_feature_type


def make_conv2d_layer(use_spec_norm,
                      spec_norm_iteration,
                      spec_norm_bound):
  """Defines type of Conv2D layer to use based on spectral normalization."""
  Conv2DBase = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=3,
      padding='same',
      use_bias=False,
      kernel_initializer='he_normal')

  def Conv2DNormed(*conv_args, **conv_kwargs):
    return ed.layers.SpectralNormalizationConv2D(
        Conv2DBase(*conv_args, **conv_kwargs),
        iteration=spec_norm_iteration,
        norm_multiplier=spec_norm_bound)

  return Conv2DNormed if use_spec_norm else Conv2DBase
# pylint: enable=invalid-name


def apply_dropout(inputs, dropout_rate, use_mc_dropout):
  """Applies a filter-wise dropout layer to the inputs."""
  logging.info('apply_dropout input shape %s', inputs.shape)
  dropout_layer = tf.keras.layers.Dropout(
      dropout_rate, noise_shape=[inputs.shape[0], 1, 1, inputs.shape[3]])

  if use_mc_dropout:
    return dropout_layer(inputs, training=True)

  return dropout_layer(inputs)


def basic_block(inputs,
                filters,
                strides,
                l2,
                use_mc_dropout,
                use_filterwise_dropout,
                dropout_rate,
                use_spec_norm,
                spec_norm_iteration,
                spec_norm_bound):
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    l2: L2 regularization coefficient.
    use_mc_dropout: Whether to apply Monte Carlo dropout.
    use_filterwise_dropout: Whether to apply filterwise dropout.
    dropout_rate: Dropout rate.
    use_spec_norm: Whether to apply spectral normalization.
    spec_norm_iteration: Number of power iterations to perform for estimating
      the spectral norm of weight matrices.
    spec_norm_bound: Upper bound to spectral norm of weight matrices.

  Returns:
    tf.Tensor.
  """
  Conv2D = make_conv2d_layer(use_spec_norm,  # pylint: disable=invalid-name
                             spec_norm_iteration,
                             spec_norm_bound)

  x = inputs
  y = inputs
  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(y)
  y = tf.keras.layers.Activation('relu')(y)
  if use_filterwise_dropout:
    y = apply_dropout(y, dropout_rate, use_mc_dropout)

  y = Conv2D(filters,
             strides=strides,
             kernel_regularizer=tf.keras.regularizers.l2(l2))(y)
  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(y)
  y = tf.keras.layers.Activation('relu')(y)
  if use_filterwise_dropout:
    y = apply_dropout(y, dropout_rate, use_mc_dropout)

  y = Conv2D(filters,
             strides=1,
             kernel_regularizer=tf.keras.regularizers.l2(l2))(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(filters,
               kernel_size=1,
               strides=strides,
               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    if use_filterwise_dropout:
      y = apply_dropout(y, dropout_rate, use_mc_dropout)

  x = tf.keras.layers.add([x, y])
  return x


def group(inputs, filters, strides, num_blocks, **kwargs):
  """Group of residual blocks."""
  x = basic_block(inputs, filters=filters, strides=strides, **kwargs)
  for _ in range(num_blocks - 1):
    x = basic_block(x, filters=filters, strides=1, **kwargs)
  return x


def wide_resnet_hetsngp(input_shape,
                        batch_size,
                        depth,
                        width_multiplier,
                        num_classes,
                        l2,
                        use_mc_dropout,
                        use_filterwise_dropout,
                        dropout_rate,
                        use_gp_layer,
                        gp_input_dim,
                        gp_hidden_dim,
                        gp_scale,
                        gp_bias,
                        gp_input_normalization,
                        gp_random_feature_type,
                        gp_cov_discount_factor,
                        gp_cov_ridge_penalty,
                        use_spec_norm,
                        spec_norm_iteration,
                        spec_norm_bound,
                        temperature,
                        num_mc_samples=100,
                        eps=1e-5,
                        sngp_var_weight=0.,
                        het_var_weight=1.):
  """Builds Wide ResNet HetSNGP.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    batch_size: The batch size of the input layer. Required by the spectral
      normalization.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    l2: L2 regularization coefficient.
    use_mc_dropout: Whether to apply Monte Carlo dropout.
    use_filterwise_dropout: Whether to apply filterwise dropout.
    dropout_rate: Dropout rate.
    use_gp_layer: Whether to use Gaussian process layer as the output layer.
    gp_input_dim: The input dimension to GP layer.
    gp_hidden_dim: The hidden dimension of the GP layer, which corresponds to
      the number of random features used for the approximation.
    gp_scale: The length-scale parameter for the RBF kernel of the GP layer.
    gp_bias: The bias term for GP layer.
    gp_input_normalization: Whether to normalize the input using LayerNorm for
      GP layer. This is similar to automatic relevance determination (ARD) in
      the classic GP learning.
    gp_random_feature_type: The type of random feature to use for
      `RandomFeatureGaussianProcess`.
    gp_cov_discount_factor: The discount factor to compute the moving average of
      precision matrix.
    gp_cov_ridge_penalty: Ridge penalty parameter for GP posterior covariance.
    use_spec_norm: Whether to apply spectral normalization.
    spec_norm_iteration: Number of power iterations to perform for estimating
      the spectral norm of weight matrices.
    spec_norm_bound: Upper bound to spectral norm of weight matrices.
    temperature: Float or scalar `Tensor` representing the softmax
      temperature.
    num_mc_samples: The number of Monte-Carlo samples used to estimate the
        predictive distribution.
    eps: Float. Clip probabilities into [eps, 1.0] softmax or
        [eps, 1.0 - eps] sigmoid before applying log (softmax), or inverse
        sigmoid.
    sngp_var_weight: Weight in [0,1] for the SNGP variance in the output.
    het_var_weight: Weight in [0,1] for the het. variance in the output.

  Returns:
    tf.keras.Model.
  """
  Conv2D = make_conv2d_layer(use_spec_norm,  # pylint: disable=invalid-name
                             spec_norm_iteration,
                             spec_norm_bound)
  OutputLayer = functools.partial(  # pylint: disable=invalid-name
      MCSoftmaxSNGP,
      num_inducing=gp_hidden_dim,
      gp_kernel_scale=gp_scale,
      gp_output_bias=gp_bias,
      normalize_input=gp_input_normalization,
      gp_cov_momentum=gp_cov_discount_factor,
      gp_cov_ridge_penalty=gp_cov_ridge_penalty,
      use_custom_random_features=True,
      custom_random_features_initializer=make_random_feature_initializer(
          gp_random_feature_type),
      temperature=temperature,
      train_mc_samples=num_mc_samples,
      test_mc_samples=num_mc_samples,
      share_samples_across_batch=True,
      logits_only=True,
      eps=eps,
      dtype=tf.float32,
      sngp_var_weight=sngp_var_weight,
      het_var_weight=het_var_weight)

  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')

  if use_mc_dropout and not use_filterwise_dropout:
    raise ValueError('cannot use mc dropout with filterwise dropout disabled.')

  num_blocks = (depth - 4) // 6
  inputs = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)

  x = Conv2D(16,
             strides=1,
             kernel_regularizer=tf.keras.regularizers.l2(l2))(inputs)
  if use_filterwise_dropout:
    x = apply_dropout(x, dropout_rate, use_mc_dropout)
  for strides, filters in zip([1, 2, 2], [16, 32, 64]):
    x = group(x,
              filters=filters * width_multiplier,
              strides=strides,
              num_blocks=num_blocks,
              l2=l2,
              use_mc_dropout=use_mc_dropout,
              use_filterwise_dropout=use_filterwise_dropout,
              dropout_rate=dropout_rate,
              use_spec_norm=use_spec_norm,
              spec_norm_iteration=spec_norm_iteration,
              spec_norm_bound=spec_norm_bound)

  x = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)

  if use_gp_layer:
    # Uses random projection to reduce the input dimension of the GP layer.
    if gp_input_dim > 0:
      x = tf.keras.layers.Dense(
          gp_input_dim,
          kernel_initializer='random_normal',
          use_bias=False,
          trainable=False)(x)
    outputs = OutputLayer(num_classes)(x)
  else:
    outputs = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        bias_regularizer=tf.keras.regularizers.l2(l2))(x)

  return tf.keras.Model(inputs=inputs, outputs=outputs)
