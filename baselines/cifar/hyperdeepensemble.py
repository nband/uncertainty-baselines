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

"""Hyper deep ensemble on CIFAR.

This script only performs evaluation, not training.
It takes as input a directory of checkpoints of models generated by some random
search. Importantly, those models are assumed to have been trained over data
not overlapping with the validation set defined here.
"""

import os

from absl import app
from absl import flags
from absl import logging

import numpy as np
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import utils  # local file import from baselines.cifar

flags.DEFINE_string('checkpoint_dir', None,
                    'The directory where the model weights are stored.'
                    'The structure is: '
                    ' checkpoint_dir/SOME_NAME_MODEL_1/.../../checkpoint*.index'
                    ' ...'
                    ' checkpoint_dir/SOME_NAME_MODEL_N/.../../checkpoint*.index'
                    'In particular, the depth of the subdir can be arbitrary.')
flags.mark_flag_as_required('checkpoint_dir')
flags.DEFINE_integer('ensemble_size', 4, 'Size of ensemble.')
flags.DEFINE_enum('greedy_objective', 'nll',
                  enum_values=['nll', 'acc', 'nll-acc'],
                  help='Objective that drives the greedy selection.')
flags.register_validator('train_proportion',
                         lambda tp: tp > 0.0 and tp <= 1.0,
                         message='--train_proportion must be in (0, 1].')
flags.FLAGS.set_default('train_proportion', 0.95)
FLAGS = flags.FLAGS


def parse_checkpoint_dir(checkpoint_dir):
  """Parse directory of checkpoints."""
  paths = []
  subdirectories = tf.io.gfile.glob(os.path.join(checkpoint_dir, '*'))
  is_checkpoint = lambda f: ('checkpoint' in f and '.index' in f)
  for subdir in subdirectories:
    for path, _, files in tf.io.gfile.walk(subdir):
      if any(f for f in files if is_checkpoint(f)):
        latest_checkpoint_without_suffix = tf.train.latest_checkpoint(path)
        paths.append(os.path.join(path, latest_checkpoint_without_suffix))
        break
  return paths


def _ensemble_accuracy(labels, logits_list):
  """Compute the accuracy resulting from the ensemble prediction."""
  per_probs = tf.nn.softmax(logits_list)
  probs = tf.reduce_mean(per_probs, axis=0)
  acc = tf.keras.metrics.SparseCategoricalAccuracy()
  acc.update_state(labels, probs)
  return acc.result()


def greedy_selection(val_logits, val_labels, max_ens_size, objective='nll'):
  """Greedy procedure from Caruana et al. 2004, with replacement."""

  assert_msg = 'Unknown objective type (received {}).'.format(objective)
  assert objective in ('nll', 'acc', 'nll-acc'), assert_msg

  if objective == 'nll':
    get_objective = lambda acc, nll: nll
  elif objective == 'acc':
    get_objective = lambda acc, nll: acc
  else:
    get_objective = lambda acc, nll: nll-acc

  best_acc = 0.
  best_nll = np.inf
  best_objective = np.inf
  ens = []

  def get_ens_size():
    return len(set(ens))

  while get_ens_size() < max_ens_size:
    current_val_logits = [val_logits[model_id] for model_id in ens]
    best_model_id = None
    for model_id, logits in enumerate(val_logits):
      acc = _ensemble_accuracy(val_labels, current_val_logits + [logits])
      negative_log_likelihood_metric = rm.metrics.EnsembleCrossEntropy()
      negative_log_likelihood_metric.add_batch(
          current_val_logits + [logits], labels=val_labels)
      nll = list(negative_log_likelihood_metric.result().values())[0]
      obj = get_objective(acc, nll)
      if obj < best_objective:
        best_acc = acc
        best_nll = nll
        best_objective = obj
        best_model_id = model_id
    if best_model_id is None:
      logging.info('Ensemble could not be improved: Greedy selection stops.')
      break
    ens.append(best_model_id)
  return ens, best_acc, best_nll


def main(argv):
  del argv  # unused arg
  if not FLAGS.use_gpu:
    raise ValueError('Only GPU is currently supported.')
  if FLAGS.num_cores > 1:
    raise ValueError('Only a single accelerator is currently supported.')
  tf.random.set_seed(FLAGS.seed)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  ds_info = tfds.builder(FLAGS.dataset).info
  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size
  num_classes = ds_info.features['label'].num_classes

  data_dir = FLAGS.data_dir
  dataset = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.TEST).load(batch_size=batch_size)
  validation_percent = 1. - FLAGS.train_proportion
  val_dataset = ub.datasets.get(
      dataset_name=FLAGS.dataset,
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.VALIDATION,
      validation_percent=validation_percent,
      drop_remainder=False).load(batch_size=batch_size)
  steps_per_val_eval = int(ds_info.splits['train'].num_examples *
                           validation_percent) // batch_size

  test_datasets = {'clean': dataset}
  if FLAGS.dataset == 'cifar100':
    data_dir = FLAGS.cifar100_c_path
  corruption_types, _ = utils.load_corrupted_test_info(FLAGS.dataset)
  for corruption_type in corruption_types:
    for severity in range(1, 6):
      dataset = ub.datasets.get(
          f'{FLAGS.dataset}_corrupted',
          corruption_type=corruption_type,
          data_dir=data_dir,
          severity=severity,
          split=tfds.Split.TEST).load(batch_size=batch_size)
      test_datasets[f'{corruption_type}_{severity}'] = dataset

  model = ub.models.wide_resnet(
      input_shape=ds_info.features['image'].shape,
      depth=28,
      width_multiplier=10,
      num_classes=num_classes,
      l2=0.)
  logging.info('Model input shape: %s', model.input_shape)
  logging.info('Model output shape: %s', model.output_shape)
  logging.info('Model number of weights: %s', model.count_params())

  # Search for checkpoints
  ensemble_filenames = parse_checkpoint_dir(FLAGS.checkpoint_dir)

  model_pool_size = len(ensemble_filenames)
  logging.info('Model pool size: %s', model_pool_size)
  logging.info('Ensemble size: %s', FLAGS.ensemble_size)
  logging.info('Ensemble number of weights: %s',
               FLAGS.ensemble_size * model.count_params())
  logging.info('Ensemble filenames: %s', str(ensemble_filenames))
  checkpoint = tf.train.Checkpoint(model=model)

  # Compute the logits on the validation set
  val_logits, val_labels = [], []
  for m, ensemble_filename in enumerate(ensemble_filenames):
    # Enforce memory clean-up
    tf.keras.backend.clear_session()
    checkpoint.restore(ensemble_filename)
    val_iterator = iter(val_dataset)
    val_logits_m = []
    for _ in range(steps_per_val_eval):
      inputs = next(val_iterator)
      features = inputs['features']
      labels = inputs['labels']
      val_logits_m.append(model(features, training=False))
      if m == 0:
        val_labels.append(labels)

    val_logits.append(tf.concat(val_logits_m, axis=0))
    if m == 0:
      val_labels = tf.concat(val_labels, axis=0)

    percent = (m + 1.) / model_pool_size
    message = ('{:.1%} completion for prediction on validation set: '
               'model {:d}/{:d}.'.format(percent, m + 1, model_pool_size))
    logging.info(message)

  selected_members, val_acc, val_nll = greedy_selection(val_logits, val_labels,
                                                        FLAGS.ensemble_size,
                                                        FLAGS.greedy_objective)
  unique_selected_members = list(set(selected_members))
  message = ('Members selected by greedy procedure: {} (with {} unique '
             'member(s))\n\t{}').format(
                 selected_members, len(unique_selected_members),
                 [ensemble_filenames[i] for i in selected_members])
  logging.info(message)
  val_metrics = {
      'val/accuracy': tf.keras.metrics.Mean(),
      'val/negative_log_likelihood': tf.keras.metrics.Mean()
  }
  val_metrics['val/accuracy'].update_state(val_acc)
  val_metrics['val/negative_log_likelihood'].update_state(val_nll)

  # Write model predictions to files.
  num_datasets = len(test_datasets)
  for m, member_id in enumerate(unique_selected_members):
    ensemble_filename = ensemble_filenames[member_id]
    checkpoint.restore(ensemble_filename)
    for n, (name, test_dataset) in enumerate(test_datasets.items()):
      filename = '{dataset}_{member}.npy'.format(dataset=name, member=member_id)
      filename = os.path.join(FLAGS.output_dir, filename)
      if not tf.io.gfile.exists(filename):
        logits = []
        test_iterator = iter(test_dataset)
        for _ in range(steps_per_eval):
          features = next(test_iterator)['features']  # pytype: disable=unsupported-operands
          logits.append(model(features, training=False))

        logits = tf.concat(logits, axis=0)
        with tf.io.gfile.GFile(filename, 'w') as f:
          np.save(f, logits.numpy())

      numerator = m * num_datasets + (n + 1)
      denominator = len(unique_selected_members) * num_datasets
      percent = numerator / denominator
      message = ('{:.1%} completion for prediction: ensemble member {:d}/{:d}. '
                 'Dataset {:d}/{:d}'.format(percent,
                                            m + 1,
                                            len(unique_selected_members),
                                            n + 1,
                                            num_datasets))
      logging.info(message)

  metrics = {
      'test/negative_log_likelihood': tf.keras.metrics.Mean(),
      'test/gibbs_cross_entropy': tf.keras.metrics.Mean(),
      'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
      'test/ece': rm.metrics.ExpectedCalibrationError(
          num_bins=FLAGS.num_bins),
      'test/diversity': rm.metrics.AveragePairwiseDiversity(),
  }
  metrics.update(val_metrics)
  corrupt_metrics = {}
  for name in test_datasets:
    corrupt_metrics['test/nll_{}'.format(name)] = tf.keras.metrics.Mean()
    corrupt_metrics['test/accuracy_{}'.format(name)] = (
        tf.keras.metrics.SparseCategoricalAccuracy())
    corrupt_metrics['test/ece_{}'.format(name)] = (
        rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins))
  for i in range(len(unique_selected_members)):
    metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
    metrics['test/accuracy_member_{}'.format(i)] = (
        tf.keras.metrics.SparseCategoricalAccuracy())

  # Evaluate model predictions.
  for n, (name, test_dataset) in enumerate(test_datasets.items()):
    logits_dataset = []
    for member_id in selected_members:
      filename = '{dataset}_{member}.npy'.format(dataset=name, member=member_id)
      filename = os.path.join(FLAGS.output_dir, filename)
      with tf.io.gfile.GFile(filename, 'rb') as f:
        logits_dataset.append(np.load(f))

    logits_dataset = tf.convert_to_tensor(logits_dataset)
    test_iterator = iter(test_dataset)
    for step in range(steps_per_eval):
      labels = next(test_iterator)['labels']  # pytype: disable=unsupported-operands
      logits = logits_dataset[:, (step*batch_size):((step+1)*batch_size)]
      labels = tf.cast(labels, tf.int32)
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

        # Attention must be paid to deal with duplicated members:
        # e.g.,
        #.    selected_members = [2, 7, 3, 3]
        #     unique_selected_members = [2, 3, 7]
        #     selected_members.index(3) --> 2
        for member_id in unique_selected_members:
          i = selected_members.index(member_id)
          member_probs = per_probs[i]
          member_loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, member_probs)
          metrics['test/nll_member_{}'.format(i)].update_state(member_loss)
          metrics['test/accuracy_member_{}'.format(i)].update_state(
              labels, member_probs)
        metrics['test/diversity'].add_batch(per_probs)
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
                                                    corruption_types)
  total_results = {name: metric.result() for name, metric in metrics.items()}
  total_results.update(corrupt_results)
  # Results from Robustness Metrics themselves return a dict, so flatten them.
  total_results = utils.flatten_dictionary(total_results)

  logging.info('Metrics: %s', total_results)


if __name__ == '__main__':
  app.run(main)
