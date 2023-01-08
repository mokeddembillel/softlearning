import sys

import numpy as np

from softlearning import algorithms
from softlearning.environments.utils import get_environment
from softlearning.misc.plotter import QFPolicyPlotter
# from softlearning.samplers import SimpleSampler
from softlearning import samplers
from softlearning import policies
from softlearning.replay_pools import SimpleReplayPool
from softlearning import value_functions
from examples.instrument import run_example_local

###############
import tree
import ray
from ray import tune
from softlearning.utils.tensorflow import set_gpu_memory_growth
import os
import copy
import glob
import pickle
import sys
import json
import tensorflow as tf
from softlearning.utils.misc import set_seed
from softlearning import replay_pools


class ExperimentRunner(tune.Trainable):
    def setup(self, variant):
        # Set the current working directory such that the local mode
        # logs into the correct place. This would not be needed on
        # local/cluster mode.
        if ray.worker._mode() == ray.worker.LOCAL_MODE:
            os.chdir(os.getcwd())

        set_seed(variant['run_params']['seed'])

        if variant['run_params'].get('run_eagerly', False):
            tf.config.experimental_run_functions_eagerly(True)

        self._variant = variant
        set_gpu_memory_growth(True)

        self.train_generator = None
        self._built = False

    def _build(self):
        variant = copy.deepcopy(self._variant)
        ######################################################
        training_environment = ( 
        get_environment('gym', 'MultiGoal', 'Default-v0', {
            'actuation_cost_coeff': 30,
            'distance_cost_coeff': 1,
            'goal_reward': 10,
            'init_sigma': 0.1,
        }))

        evaluation_environment = training_environment.copy()



        variant['Q_params']['config'].update({
        'input_shapes': (
            training_environment.observation_shape,
            training_environment.action_shape,
        )
        })
        # Qs = value_functions.get(variant['Q_params'])
        Qs = self.Qs = tree.flatten(value_functions.get(variant['Q_params']))


        variant['policy_params']['config'].update({
            'action_range': (training_environment.action_space.low,
                            training_environment.action_space.high),
            'input_shapes': training_environment.observation_shape,
            'output_shape': training_environment.action_shape,
        })
        # policy = policies.get(variant['policy_params'])
        policy = self.policy = policies.get(variant['policy_params'])
        
        
        replay_pool = self.replay_pool = SimpleReplayPool(
            environment=training_environment,
            max_size=1e6)

        variant['sampler_params']['config'].update({
                'environment': training_environment,
                'policy': policy,
                'pool': replay_pool,
            })
        # import pdb; pdb.set_trace() ################################################

        # sampler = samplers.get(variant['sampler_params'])
        sampler = self.sampler = samplers.get(variant['sampler_params'])

        plotter = self.plotter = QFPolicyPlotter(
        Q=Qs[0],
        policy=policy,
        obs_lst=np.array(((-2.5, 0.0),
                          (0.0, 0.0),
                          (2.5, 2.5),
                          (-2.5, -2.5))),
        default_action=(np.nan, np.nan),
        n_samples=100)

        variant['algorithm_params']['config'].update({
            'training_environment': training_environment,
            'evaluation_environment': evaluation_environment,
            'policy': policy,
            'Qs': Qs,
            'pool': replay_pool,
            'sampler': sampler,
            'min_pool_size': 100,
            'batch_size': 64,
            'plotter': plotter,
        })
        # algorithm = algorithms.get(variant['algorithm_params'])
        self.algorithm = algorithms.get(variant['algorithm_params'])

        self._built = True
        ######################################################


        # variant['replay_pool_params']['config'].update({
        #     'environment': training_environment,
        # })
        # replay_pool = self.replay_pool = replay_pools.get(
        #     variant['replay_pool_params'])

    def step(self):
        if not self._built:
            self._build()

        if self.train_generator is None:
            self.train_generator = self.algorithm.train()

        diagnostics = next(self.train_generator)

        return diagnostics

    @staticmethod
    def _pickle_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'checkpoint.pkl')

    @staticmethod
    def _algorithm_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'algorithm')

    @staticmethod
    def _replay_pool_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'replay_pool.pkl')

    @staticmethod
    def _sampler_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'sampler.pkl')

    @staticmethod
    def _policy_save_path(checkpoint_dir):
        return os.path.join(checkpoint_dir, 'policy')

    def _save_replay_pool(self, checkpoint_dir):
        if not self._variant['run_params'].get(
                'checkpoint_replay_pool', False):
            return

        replay_pool_save_path = self._replay_pool_save_path(checkpoint_dir)
        self.replay_pool.save_latest_experience(replay_pool_save_path)

    def _restore_replay_pool(self, current_checkpoint_dir):
        if not self._variant['run_params'].get(
                'checkpoint_replay_pool', False):
            return

        experiment_root = os.path.dirname(current_checkpoint_dir)

        experience_paths = [
            self._replay_pool_save_path(checkpoint_dir)
            for checkpoint_dir in sorted(glob.iglob(
                os.path.join(experiment_root, 'checkpoint_*')))
        ]

        for experience_path in experience_paths:
            self.replay_pool.load_experience(experience_path)

    def _save_sampler(self, checkpoint_dir):
        with open(self._sampler_save_path(checkpoint_dir), 'wb') as f:
            pickle.dump(self.sampler, f)

    def _restore_sampler(self, checkpoint_dir):
        with open(self._sampler_save_path(checkpoint_dir), 'rb') as f:
            sampler = pickle.load(f)

        self.sampler.__setstate__(sampler.__getstate__())
        self.sampler.initialize(
            self.training_environment, self.policy, self.replay_pool)

    def _save_value_functions(self, checkpoint_dir):
        tree.map_structure_with_path(
            lambda path, Q: Q.save_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('Q', *[str(x) for x in path]))),
                save_format='tf'),
            self.Qs)

    def _restore_value_functions(self, checkpoint_dir):
        tree.map_structure_with_path(
            lambda path, Q: Q.load_weights(
                os.path.join(
                    checkpoint_dir, '-'.join(('Q', *[str(x) for x in path])))),
            self.Qs)

    def _save_policy(self, checkpoint_dir):
        save_path = self._policy_save_path(checkpoint_dir)
        self.policy.save(save_path)

    def _restore_policy(self, checkpoint_dir):
        save_path = self._policy_save_path(checkpoint_dir)
        status = self.policy.load_weights(save_path)
        status.assert_consumed().run_restore_ops()

    def _save_algorithm(self, checkpoint_dir):
        save_path = self._algorithm_save_path(checkpoint_dir)

        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)
        tf_checkpoint.save(file_prefix=f"{save_path}/checkpoint")

        state = self.algorithm.__getstate__()
        with open(os.path.join(save_path, "state.json"), 'w') as f:
            json.dump(state, f)

    def _restore_algorithm(self, checkpoint_dir):
        save_path = self._algorithm_save_path(checkpoint_dir)

        with open(os.path.join(save_path, "state.json"), 'r') as f:
            state = json.load(f)

        self.algorithm.__setstate__(state)

        # NOTE(hartikainen): We need to run one step on optimizers s.t. the
        # variables get initialized.
        # TODO(hartikainen): This should be done somewhere else.
        tree.map_structure(
            lambda Q_optimizer, Q: Q_optimizer.apply_gradients([
                (tf.zeros_like(variable), variable)
                for variable in Q.trainable_variables
            ]),
            tuple(self.algorithm._Q_optimizers),
            tuple(self.Qs),
        )

        self.algorithm._alpha_optimizer.apply_gradients([(
            tf.zeros_like(self.algorithm._log_alpha), self.algorithm._log_alpha
        )])
        self.algorithm._policy_optimizer.apply_gradients([
            (tf.zeros_like(variable), variable)
            for variable in self.policy.trainable_variables
        ])

        tf_checkpoint = tf.train.Checkpoint(**self.algorithm.tf_saveables)

        status = tf_checkpoint.restore(tf.train.latest_checkpoint(
            # os.path.split(f"{save_path}/checkpoint")[0])
            # f"{save_path}/checkpoint-xxx"))
            os.path.split(os.path.join(save_path, "checkpoint"))[0]))
        status.assert_consumed().run_restore_ops()

    def save_checkpoint(self, checkpoint_dir):
        """Implements the checkpoint save logic."""
        self._save_replay_pool(checkpoint_dir)
        self._save_sampler(checkpoint_dir)
        self._save_value_functions(checkpoint_dir)
        self._save_policy(checkpoint_dir)
        self._save_algorithm(checkpoint_dir)

        return os.path.join(checkpoint_dir, '')

    def load_checkpoint(self, checkpoint_dir):
        """Implements the checkpoint restore logic."""
        assert isinstance(checkpoint_dir, str), checkpoint_dir
        checkpoint_dir = checkpoint_dir.rstrip('/')

        self._build()

        self._restore_replay_pool(checkpoint_dir)
        self._restore_sampler(checkpoint_dir)
        self._restore_value_functions(checkpoint_dir)
        self._restore_policy(checkpoint_dir)
        self._restore_algorithm(checkpoint_dir)

        for Q, Q_target in zip(self.algorithm._Qs, self.algorithm._Q_targets):
            Q_target.set_weights(Q.get_weights())

        self._built = True

#########################################################################################
#########################################################################################

# def run_experiment(variant, reporter):
#     training_environment = (
#         get_environment('gym', 'MultiGoal', 'Default-v0', {
#             'actuation_cost_coeff': 30,
#             'distance_cost_coeff': 1,
#             'goal_reward': 10,
#             'init_sigma': 0.1,
#         }))

#     evaluation_environment = training_environment.copy()

#     pool = SimpleReplayPool(
#         environment=training_environment,
#         max_size=1e6)


#     variant['Q_params']['config'].update({
#         'input_shapes': (
#             training_environment.observation_shape,
#             training_environment.action_shape,
#         )
#     })
#     Qs = value_functions.get(variant['Q_params'])

#     variant['policy_params']['config'].update({
#         'action_range': (training_environment.action_space.low,
#                          training_environment.action_space.high),
#         'input_shapes': training_environment.observation_shape,
#         'output_shape': training_environment.action_shape,
#     })
#     policy = policies.get(variant['policy_params'])

#     variant['sampler_params']['config'].update({
#             'environment': training_environment,
#             'policy': policy,
#             'pool': pool,
#         })
#     # import pdb; pdb.set_trace() ################################################

#     sampler = samplers.get(variant['sampler_params'])

#     plotter = QFPolicyPlotter(
#         Q=Qs[0],
#         policy=policy,
#         obs_lst=np.array(((-2.5, 0.0),
#                           (0.0, 0.0),
#                           (2.5, 2.5),
#                           (-2.5, -2.5))),
#         default_action=(np.nan, np.nan),
#         n_samples=100)

#     variant['algorithm_params']['config'].update({
#         'training_environment': training_environment,
#         'evaluation_environment': evaluation_environment,
#         'policy': policy,
#         'Qs': Qs,
#         'pool': pool,
#         'sampler': sampler,
#         'min_pool_size': 100,
#         'batch_size': 64,
#         'plotter': plotter,
#     })
#     algorithm = algorithms.get(variant['algorithm_params'])

#     for train_result in algorithm.train():
#         reporter(**train_result)


def main(argv=None):
    """Run ExperimentRunner locally on ray.

    To run this example on cloud (e.g. gce/ec2), use the setup scripts:
    'softlearning launch_example_{gce,ec2} examples.development <options>'.

    Run 'softlearning launch_example_{gce,ec2} --help' for further
    instructions.
    """
    run_example_local('examples.multi_goal', argv)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
