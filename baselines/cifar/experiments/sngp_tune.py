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

r"""SNGP for WideResNet CIFAR-10.

"""

import datetime
import getpass
import os.path

from ml_collections import config_dict


def get_config():
  """Returns the configuration for this experiment."""
  config = config_dict.ConfigDict()
  config.user = getpass.getuser()
  config.priority = 'prod'
  config.platform = 'gpu'
  config.gpu_type = 't4'
  config.num_gpus = 1
  config.experiment_name = (
      os.path.splitext(os.path.basename(__file__))[0] + '_' +
      datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
  output_dir = 'gs://launcher-beta-test-bucket/{}'.format(
      config.experiment_name)
  config.args = {
      'base_learning_rate': 0.08,
      'gp_mean_field_factor': 20,  # 25,
      'train_epochs': 250,
      'per_core_batch_size': 64,
      'data_dir': output_dir,
      'output_dir': output_dir,
      'download_data': True,
      'train_proportion': 0.9,
      'eval_on_ood': True,
      'ood_dataset': 'cifar100,svhn_cropped',
      # If drop_remainder=false, it will cause the issue of
      # `TPU has inputs with dynamic shapes`
      'drop_remainder_for_eval': True,
      'corruptions_interval': 10,
  }
  return config


def get_sweep(hyper):
  return hyper.product([
      hyper.sweep('seed', list(range(10))),
      hyper.sweep('use_spec_norm', hyper.categorical([False, True])),
      hyper.sweep('use_gp_layer', hyper.categorical([False, True])),
      hyper.sweep('dempster_shafer_ood', hyper.categorical([False, True])),
  ])
