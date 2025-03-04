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

"""Binary to train a deterministic MPNN model."""
import collections
import dataclasses
import logging
import os
import time
from typing import Dict, Sequence

from absl import app
from absl import flags
import robustness_metrics as rm
import tensorflow as tf
from tensorflow_addons import losses as tfa_losses
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import utils  # local file import from baselines.drug_cardiotoxicity

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'data_dir', None,
    'Directory containing the TFRecord datasets for Drug Cardiotoxicity.')
flags.DEFINE_integer('num_heads', 2, 'Number of classification heads.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_string('job_base_dir', None,
                    'Output directory for the umbrella job, which may have '
                    'multilple models from ensemble strategy or Vizier.'
                    'It can be set to None when model_dir is specified.'
                    'When we use Vizier and there is no model_dir specified, '
                    'this flag should be specified.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('num_epochs', 200, 'Number of epochs.')
flags.DEFINE_boolean('use_gpu', True, 'If True, uses GPU.')
flags.DEFINE_integer('num_cores', 1, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', '', 'TPU master.')
# Parameter flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('num_layers', 2, 'Number of message passing layers.')
flags.DEFINE_integer('message_layer_size', 32,
                     'Number of units in message layers.')
flags.DEFINE_integer('readout_layer_size', 32,
                     'Number of units in the readout layer.')

# Loss type.
flags.DEFINE_enum('loss_type', 'xent', ['xent', 'focal'],
                  'Type of loss function to use.')


def run(
    train_dataset: tf.data.Dataset,
    eval_datasets: Dict[str, tf.data.Dataset],
    steps_per_eval: Dict[str, int],
    params: utils.ModelParameters,
    model_dir: str,
    strategy: tf.distribute.Strategy,
    summary_writer: tf.summary.SummaryWriter,
    loss_type: str):
  """Trains and evaluates the model."""
  with strategy.scope():
    model = ub.models.mpnn(
        nodes_shape=train_dataset.element_spec[0]['atoms'].shape[1:],
        edges_shape=train_dataset.element_spec[0]['pairs'].shape[1:],
        num_heads=params.num_heads,
        num_layers=params.num_layers,
        message_layer_size=params.message_layer_size,
        readout_layer_size=params.readout_layer_size,
        use_gp_layer=params.use_gp_layer)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=params.learning_rate)
    metrics = {
        'train/negative_log_likelihood': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.CategoricalAccuracy(),
        'train/loss': tf.keras.metrics.Mean(),
        'train/roc_auc': tf.keras.metrics.AUC(),
    }

    for dataset_name in eval_datasets:
      metrics[
          f'{dataset_name}/accuracy'] = tf.keras.metrics.CategoricalAccuracy()
      metrics[f'{dataset_name}/roc_auc'] = tf.keras.metrics.AUC()
      metrics[
          f'{dataset_name}/negative_log_likelihood'] = tf.keras.metrics.Mean()
      if dataset_name == 'test2':
        ece_num_bins = 5
      else:
        ece_num_bins = 10
      metrics[f'{dataset_name}/ece'] = rm.metrics.ExpectedCalibrationError(
          num_bins=ece_num_bins)
      metrics[f'{dataset_name}/brier'] = rm.metrics.Brier()

  @tf.function
  def train_step(iterator):
    """Training StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      if len(inputs) == 3:
        features, labels, sample_weights = inputs
      else:
        features, labels = inputs
        sample_weights = 1

      with tf.GradientTape() as tape:
        probs = model(features, training=True)
        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(labels, probs) *
            sample_weights)

        l2_loss = sum(model.losses)
        if loss_type == 'focal':
          focal_loss_fn = tfa_losses.SigmoidFocalCrossEntropy()
          focal_loss = tf.reduce_mean(
              focal_loss_fn(labels, probs) * sample_weights)
          loss = focal_loss + l2_loss
        else:
          loss = negative_log_likelihood + l2_loss
        # Scale the loss given the tf.distribute.Strategy will reduce sum all
        # gradients. See details in
        # https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, probs)
      metrics['train/roc_auc'].update_state(labels[:, 1], probs[:, 1])

    for _ in tf.range(tf.cast(params.steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def eval_step(iterator, dataset_name, num_steps):
    """Evaluation StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      if len(inputs) == 3:
        features, labels, _ = inputs
      else:
        features, labels = inputs

      probs = model(features, training=False)
      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.categorical_crossentropy(labels, probs))

      metrics[f'{dataset_name}/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics[f'{dataset_name}/accuracy'].update_state(labels, probs)
      metrics[f'{dataset_name}/roc_auc'].update_state(
          labels[:, 1], probs[:, 1])
      metrics[f'{dataset_name}/ece'].add_batch(probs[:, 1], label=labels[:, 1])
      metrics[f'{dataset_name}/brier'].add_batch(probs, label=labels[:, 1])

    for _ in tf.range(tf.cast(num_steps, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  # Makes datasets into distributed version.
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  eval_datasets = {
      ds_name: strategy.experimental_distribute_dataset(ds)
      for ds_name, ds in eval_datasets.items()
  }
  logging.info('Number of replicas in sync: %s', strategy.num_replicas_in_sync)

  train_iterator = iter(train_dataset)
  start_time = time.time()
  metrics_history = collections.defaultdict(list)
  for epoch in range(params.num_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    train_step(train_iterator)

    current_step = (epoch + 1) * params.steps_per_epoch
    max_steps = params.steps_per_epoch * params.num_epochs
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   current_step / max_steps, epoch + 1, params.num_epochs,
                   steps_per_sec, eta_seconds / 60, time_elapsed / 60))
    logging.info(message)

    # Start evaluation.
    logging.info('Starting to run eval at epoch: %s', epoch)
    for dataset_name, eval_dataset in eval_datasets.items():
      eval_iterator = iter(eval_dataset)
      eval_step(eval_iterator, dataset_name, steps_per_eval[dataset_name])

    metrics_history['epoch'].append(epoch + 1)
    with summary_writer.as_default():
      for name, metric in metrics.items():
        result = utils.get_metric_result_value(metric)
        tf.summary.scalar(name, result, step=epoch + 1)
        metrics_history[name].append(str(result))


    for metric in metrics.values():
      metric.reset_states()

    model.save(os.path.join(model_dir, f'model_{epoch + 1}'), overwrite=True)

  utils.write_params(metrics_history,
                     os.path.join(model_dir, 'metrics_history.json'))


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  if not FLAGS.use_gpu:
    logging.info('Using TPU for training.')
    strategy = utils.get_tpu_strategy(FLAGS.tpu)
  else:
    logging.info('Using GPU for training.')
    strategy = tf.distribute.MirroredStrategy()

  train_dataset, steps_per_epoch = utils.load_dataset(FLAGS.data_dir,
                                                      tfds.Split.TRAIN,
                                                      FLAGS.batch_size)

  eval_identifiers = ['tune', 'test1', 'test2']
  splits = [tfds.Split.VALIDATION, tfds.Split.TEST, tfds.Split('test2')]
  eval_datasets, steps_per_eval = utils.load_eval_datasets(
      eval_identifiers, splits, FLAGS.data_dir, FLAGS.batch_size)

  logging.info('Steps for eval datasets: %s', steps_per_eval)

  params = utils.ModelParameters(
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      message_layer_size=FLAGS.message_layer_size,
      readout_layer_size=FLAGS.readout_layer_size,
      use_gp_layer=False,
      learning_rate=FLAGS.learning_rate,
      num_epochs=FLAGS.num_epochs,
      steps_per_epoch=steps_per_epoch)

  model_dir = FLAGS.output_dir
  utils.write_params(
      dataclasses.asdict(params), os.path.join(model_dir, 'params.json'))

  summary_writer = tf.summary.create_file_writer(
      os.path.join(model_dir, 'summaries'))
  run(train_dataset=train_dataset,
      eval_datasets=eval_datasets,
      steps_per_eval=steps_per_eval,
      params=params,
      model_dir=model_dir,
      strategy=strategy,
      summary_writer=summary_writer,
      loss_type=FLAGS.loss_type)


if __name__ == '__main__':
  app.run(main)
