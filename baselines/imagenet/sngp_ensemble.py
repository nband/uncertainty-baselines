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

"""Ensemble of SNGP models on ImageNet."""

import os

from absl import app
from absl import flags
from absl import logging

import edward2 as ed
import numpy as np
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import utils  # local file import from baselines.imagenet

flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_string('checkpoint_dir', None,
                    'The directory where the model weights are stored.')
flags.mark_flag_as_required('checkpoint_dir')
flags.DEFINE_string('output_dir', '/tmp/imagenet',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_string('alexnet_errors_path', None,
                    'Path to AlexNet corruption errors file.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE computation.')

# SNGP ensemble flags
flags.DEFINE_float(
    'gp_mean_field_factor_ensemble', -1,
    'The tunable multiplicative factor used in the mean-field approximation '
    'for the posterior mean of softmax Gaussian process. If -1 then use '
    'posterior mode instead of posterior mean.')

# Dropout flags.
flags.DEFINE_bool('use_mc_dropout', False,
                  'Whether to use Monte Carlo dropout during inference.')
flags.DEFINE_float('dropout_rate', 0., 'Dropout rate.')
flags.DEFINE_bool(
    'filterwise_dropout', True, 'Dropout whole convolutional'
    'filters instead of individual values in the feature map.')

# Spectral normalization flags.
flags.DEFINE_bool('use_spec_norm', True,
                  'Whether to apply spectral normalization.')
flags.DEFINE_integer(
    'spec_norm_iteration', 1,
    'Number of power iterations to perform for estimating '
    'the spectral norm of weight matrices.')
flags.DEFINE_float('spec_norm_bound', 6.,
                   'Upper bound to spectral norm of weight matrices.')

# Gaussian process flags.
flags.DEFINE_bool('use_gp_layer', True,
                  'Whether to use Gaussian process as the output layer.')
flags.DEFINE_float('gp_bias', 0., 'The bias term for GP layer.')
flags.DEFINE_float(
    'gp_scale', 1.,
    'The length-scale parameter for the RBF kernel of the GP layer.')
flags.DEFINE_integer(
    'gp_hidden_dim', 1024,
    'The hidden dimension of the GP layer, which corresponds to the number of '
    'random features used for the approximation.')
flags.DEFINE_bool(
    'gp_input_normalization', False,
    'Whether to normalize the input for GP layer using LayerNorm. This is '
    'similar to applying automatic relevance determination (ARD) in the '
    'classic GP literature.')
flags.DEFINE_string(
    'gp_random_feature_type', 'orf',
    'The type of random feature to use. One of "rff" (random Fourier feature), '
    '"orf" (orthogonal random feature).')
flags.DEFINE_float('gp_cov_ridge_penalty', 1.,
                   'Ridge penalty parameter for GP posterior covariance.')
flags.DEFINE_float(
    'gp_cov_discount_factor', -1,
    'The discount factor to compute the moving average of precision matrix.'
    'If -1 then instead compute the exact covariance at the lastest epoch.')
flags.DEFINE_bool(
    'gp_output_imagenet_initializer', True,
    'Whether to initialize GP output layer using Gaussian with small '
    'standard deviation (sd=0.01).')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_integer('num_cores', 1, 'Number of TPU cores or number of GPUs.')

FLAGS = flags.FLAGS

# Number of images in eval dataset.
IMAGENET_VALIDATION_IMAGES = 50000
NUM_CLASSES = 1000


def main(argv):
  del argv  # unused arg
  if not FLAGS.use_gpu:
    raise ValueError('Only GPU is currently supported.')
  if FLAGS.num_cores > 1:
    raise ValueError('Only a single accelerator is currently supported.')
  tf.random.set_seed(FLAGS.seed)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  steps_per_eval = IMAGENET_VALIDATION_IMAGES // batch_size

  test_builder = ub.datasets.ImageNetDataset(
      split=tfds.Split.TEST,
      use_bfloat16=FLAGS.use_bfloat16,
      data_dir=FLAGS.data_dir)
  clean_test_dataset = test_builder.load(batch_size=batch_size)
  test_datasets = {'clean': clean_test_dataset}
  corruption_types, max_intensity = utils.load_corrupted_test_info()
  for name in corruption_types:
    for intensity in range(1, max_intensity + 1):
      dataset_name = '{0}_{1}'.format(name, intensity)
      test_datasets[dataset_name] = utils.load_corrupted_test_dataset(
          corruption_name=name,
          corruption_intensity=intensity,
          batch_size=batch_size,
          drop_remainder=True,
          use_bfloat16=False)

  model = ub.models.resnet50_sngp(
      input_shape=(224, 224, 3),
      batch_size=FLAGS.per_core_batch_size,
      num_classes=NUM_CLASSES,
      use_mc_dropout=FLAGS.use_mc_dropout,
      dropout_rate=FLAGS.dropout_rate,
      filterwise_dropout=FLAGS.filterwise_dropout,
      use_gp_layer=FLAGS.use_gp_layer,
      gp_hidden_dim=FLAGS.gp_hidden_dim,
      gp_scale=FLAGS.gp_scale,
      gp_bias=FLAGS.gp_bias,
      gp_input_normalization=FLAGS.gp_input_normalization,
      gp_random_feature_type=FLAGS.gp_random_feature_type,
      gp_cov_discount_factor=FLAGS.gp_cov_discount_factor,
      gp_cov_ridge_penalty=FLAGS.gp_cov_ridge_penalty,
      gp_output_imagenet_initializer=FLAGS.gp_output_imagenet_initializer,
      use_spec_norm=FLAGS.use_spec_norm,
      spec_norm_iteration=FLAGS.spec_norm_iteration,
      spec_norm_bound=FLAGS.spec_norm_bound)

  logging.info('Model input shape: %s', model.input_shape)
  logging.info('Model output shape: %s', model.output_shape)
  logging.info('Model number of weights: %s', model.count_params())
  # Search for checkpoints from their index file; then remove the index suffix.
  ensemble_filenames = tf.io.gfile.glob(os.path.join(FLAGS.checkpoint_dir,
                                                     '**/*.index'))
  ensemble_filenames = [filename[:-6] for filename in ensemble_filenames]
  ensemble_size = len(ensemble_filenames)
  logging.info('Ensemble size: %s', ensemble_size)
  logging.info('Ensemble number of weights: %s',
               ensemble_size * model.count_params())
  logging.info('Ensemble filenames: %s', str(ensemble_filenames))
  checkpoint = tf.train.Checkpoint(model=model)

  # Write model predictions to files.
  num_datasets = len(test_datasets)
  for m, ensemble_filename in enumerate(ensemble_filenames):
    checkpoint.restore(ensemble_filename)
    for n, (name, test_dataset) in enumerate(test_datasets.items()):
      filename = '{dataset}_{member}.npy'.format(dataset=name, member=m)
      filename = os.path.join(FLAGS.output_dir, filename)
      if not tf.io.gfile.exists(filename):
        logits = []
        test_iterator = iter(test_dataset)
        for _ in range(steps_per_eval):
          features, _ = next(test_iterator)  # pytype: disable=attribute-error
          logits_member, covmat_member = model(features, training=False)
          logits_member = ed.layers.utils.mean_field_logits(
              logits_member, covmat_member, FLAGS.gp_mean_field_factor_ensemble)
          logits.append(logits_member)

        logits = tf.concat(logits, axis=0)
        with tf.io.gfile.GFile(filename, 'w') as f:
          np.save(f, logits.numpy())
      percent = (m * num_datasets + (n + 1)) / (ensemble_size * num_datasets)
      message = ('{:.1%} completion for prediction: ensemble member {:d}/{:d}. '
                 'Dataset {:d}/{:d}'.format(percent,
                                            m + 1,
                                            ensemble_size,
                                            n + 1,
                                            num_datasets))
      logging.info(message)

  metrics = {
      'test/negative_log_likelihood': tf.keras.metrics.Mean(),
      'test/gibbs_cross_entropy': tf.keras.metrics.Mean(),
      'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
      'test/ece': rm.metrics.ExpectedCalibrationError(
          num_bins=FLAGS.num_bins),
  }
  corrupt_metrics = {}
  for name in test_datasets:
    corrupt_metrics['test/nll_{}'.format(name)] = tf.keras.metrics.Mean()
    corrupt_metrics['test/accuracy_{}'.format(name)] = (
        tf.keras.metrics.SparseCategoricalAccuracy())
    corrupt_metrics['test/ece_{}'.format(
        name)] = rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins)

  # Evaluate model predictions.
  for n, (name, test_dataset) in enumerate(test_datasets.items()):
    logits_dataset = []
    for m in range(ensemble_size):
      filename = '{dataset}_{member}.npy'.format(dataset=name, member=m)
      filename = os.path.join(FLAGS.output_dir, filename)
      with tf.io.gfile.GFile(filename, 'rb') as f:
        logits_dataset.append(np.load(f))

    logits_dataset = tf.convert_to_tensor(logits_dataset)
    test_iterator = iter(test_dataset)
    for step in range(steps_per_eval):
      _, labels = next(test_iterator)  # pytype: disable=attribute-error
      logits = logits_dataset[:, (step*batch_size):((step+1)*batch_size)]
      labels = tf.cast(tf.reshape(labels, [-1]), tf.int32)
      negative_log_likelihood_metric = rm.metrics.EnsembleCrossEntropy()
      negative_log_likelihood_metric.add_batch(logits, labels=labels)
      negative_log_likelihood = list(
          negative_log_likelihood_metric.result().values())[0]
      per_probs = tf.nn.softmax(logits)
      probs = tf.reduce_mean(per_probs, axis=0)
      if name == 'clean':
        gibbs_ce_metric = rm.metrics.GibbsCrossEntropy()
        gibbs_ce_metric.add_batch(logits, labels=labels)
        gibbs_ce = list(gibbs_ce_metric.result().values())[0]
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/gibbs_cross_entropy'].update_state(gibbs_ce)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].add_batch(probs, label=labels)
      else:
        corrupt_metrics['test/nll_{}'.format(name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test/accuracy_{}'.format(name)].update_state(
            labels, probs)
        corrupt_metrics['test/ece_{}'.format(name)].add_batch(
            probs, label=labels)

    message = ('{:.1%} completion for evaluation: dataset {:d}/{:d}'.format(
        (n + 1) / num_datasets, n + 1, num_datasets))
    logging.info(message)

  corrupt_results = utils.aggregate_corrupt_metrics(corrupt_metrics,
                                                    corruption_types,
                                                    max_intensity,
                                                    FLAGS.alexnet_errors_path)
  total_results = {name: metric.result() for name, metric in metrics.items()}
  # Metrics from Robustness Metrics (like ECE) will return a dict with a
  # single key/value, instead of a scalar.
  total_results.update(corrupt_results)
  total_results = {
      k: (list(v.values())[0] if isinstance(v, dict) else v)
      for k, v in total_results.items()
  }
  logging.info('Metrics: %s', total_results)


if __name__ == '__main__':
  app.run(main)
