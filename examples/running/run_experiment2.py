#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from benchmarl.algorithms import IqlConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from pathlib import Path

if __name__ == "__main__":
    yaml_path = 'C:\\Users\giova\Documents\Research\BenchMARL\\benchmarl\conf\experiment\my_base_experiment_10env.yaml'

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment_config = ExperimentConfig.get_from_yaml(yaml_path)

    # Loads from "benchmarl/conf/task/vmas/navigation.yaml"
    task = VmasTask.NAVIGATION.get_from_yaml()

    # Loads from "benchmarl/conf/algorithm/mappo.yaml"
    algorithm_config = IqlConfig.get_from_yaml()

    # Loads from "benchmarl/conf/model/layers/mlp.yaml"
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment.run()
