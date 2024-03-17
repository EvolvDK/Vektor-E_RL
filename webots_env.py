import logging
import pickle
import socket
import subprocess
import time

import gymnasium
from gymnasium import Env
from gymnasium.wrappers import FlattenObservation

from tianshou.env import VectorEnvNormObs, BaseVectorEnv
from tianshou.highlevel.env import (
    ContinuousEnvironments,
    EnvFactoryRegistered,
    EnvPoolFactory,
    VectorEnvType, EnvMode,
)
from tianshou.highlevel.persistence import Persistence, PersistEvent, RestoreEvent
from tianshou.highlevel.world import World

envpool_is_available = True
try:
    import envpool
except ImportError:
    envpool_is_available = False
    envpool = None

log = logging.getLogger(__name__)


def make_webots_env(task: str, seed: int, num_train_envs: int, num_test_envs: int, obs_norm: bool):
    """Wrapper function for Webots env.

    If EnvPool is installed, it will automatically switch to EnvPool's Mujoco env.

    :return: a tuple of (single env, training envs, test envs).
    """
    envs = WebotsEnvFactory(task, seed, obs_norm=obs_norm).create_envs(
        num_train_envs,
        num_test_envs,
    )
    return envs.env, envs.train_envs, envs.test_envs


class WebotsEnvObsRmsPersistence(Persistence):
    FILENAME = "env_obs_rms.pkl"

    def persist(self, event: PersistEvent, world: World) -> None:
        if event != PersistEvent.PERSIST_POLICY:
            return
        obs_rms = world.envs.train_envs.get_obs_rms()
        path = world.persist_path(self.FILENAME)
        log.info(f"Saving environment obs_rms value to {path}")
        with open(path, "wb") as f:
            pickle.dump(obs_rms, f)

    def restore(self, event: RestoreEvent, world: World):
        if event != RestoreEvent.RESTORE_POLICY:
            return
        path = world.restore_path(self.FILENAME)
        log.info(f"Restoring environment obs_rms value from {path}")
        with open(path, "rb") as f:
            obs_rms = pickle.load(f)
        world.envs.train_envs.set_obs_rms(obs_rms)
        world.envs.test_envs.set_obs_rms(obs_rms)


class WebotsEnvFactory(EnvFactoryRegistered):
    def __init__(self, task: str, seed: int, obs_norm=True) -> None:
        super().__init__(
            task=task,
            seed=seed,
            venv_type=VectorEnvType.DUMMY,
            envpool_factory=EnvPoolFactory() if envpool_is_available else None,
        )
        self.obs_norm = obs_norm

    def create_envs(self, num_training_envs: int, num_test_envs: int) -> ContinuousEnvironments:
        envs = super().create_envs(num_training_envs, num_test_envs)
        assert isinstance(envs, ContinuousEnvironments)

        # obs norm wrapper
        if self.obs_norm:
            envs.train_envs = VectorEnvNormObs(envs.train_envs)
            envs.test_envs = VectorEnvNormObs(envs.test_envs, update_obs_rms=False)
            envs.test_envs.set_obs_rms(envs.train_envs.get_obs_rms())
            envs.set_persistence(WebotsEnvObsRmsPersistence())

        return envs

# class WebotsEnvFactory(EnvFactoryRegistered):
#     def __init__(self, task: str, seed: int, base_port: int, obs_norm=True, **kwargs) -> None:
#         super().__init__(
#             task=task,
#             seed=seed,
#             venv_type=VectorEnvType.SUBPROC,
#             envpool_factory=EnvPoolFactory() if envpool_is_available else None,
#             **kwargs,
#         )
#         self.base_port = base_port
#         self.current_port = base_port
#         self.obs_norm = obs_norm
#
#     def _create_kwargs(self, mode: EnvMode) -> dict:
#         """Overrides the EnvFactoryRegistered method to include port information.
#
#         :param mode: the mode for which to create the environment (TRAIN, TEST, WATCH)
#         :return: a dictionary of keyword arguments including the port
#         """
#         # Call the superclass method to get the base kwargs
#         kwargs = super()._create_kwargs(mode)
#
#         # Ensure a unique port is allocated for each environment instance
#         if not self.is_port_available(self.current_port):
#             self.current_port += 1  # Adjust this logic as needed to ensure port availability
#
#         # Include the port information in the kwargs
#         kwargs["port"] = self.current_port
#
#         # Increment the port for the next environment
#         self.current_port += 1
#
#         return kwargs
#
#     def is_port_available(self, port):
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
#             try:
#                 sock.bind(('', port))
#                 return True
#             except socket.error:
#                 return False
#
#     def create_envs(self, num_training_envs: int, num_test_envs: int) -> ContinuousEnvironments:
#         envs = super().create_envs(num_training_envs, num_test_envs)
#         assert isinstance(envs, ContinuousEnvironments)
#
#         # obs norm wrapper
#         if self.obs_norm:
#             envs.train_envs = VectorEnvNormObs(envs.train_envs)
#             envs.test_envs = VectorEnvNormObs(envs.test_envs, update_obs_rms=False)
#             envs.test_envs.set_obs_rms(envs.train_envs.get_obs_rms())
#             envs.set_persistence(WebotsEnvObsRmsPersistence())
#
#         return envs
#
#     def create_env(self, mode: EnvMode) -> Env:
#         # Get the kwargs with port information for the current mode
#         kwargs = self._create_kwargs(mode)
#
#         # Extract the port from kwargs to use in launching the Webots instance
#         port = kwargs.get("port")
#
#         if port is not None:
#             # Launch the Webots instance with the specified port
#             self.launch_webots_instance("/home/elouarn/WebotsProjects/Simulateur_CoVAPSy_Webots2023b_Base2/worlds"
#                                         "/Piste_CoVAPSy_2023b.wbt", port)
#             # Ensure the environment is aware of its specific port by adding it to the environment's constructor
#             # This might involve modifying how you register and instantiate your environments with Gymnasium
#             # to allow passing the port as a parameter
#             env = gymnasium.make(self.task, **kwargs)
#         else:
#             # Handle the case where no port is provided or found
#             raise ValueError("No port specified for Webots environment creation.")
#
#         return env

# import warnings
#
# import gymnasium as gym
#
# from tianshou.env import ShmemVectorEnv, VectorEnvNormObs, DummyVectorEnv
#
# try:
#     import envpool
# except ImportError:
#     envpool = None
#
#
# def make_webots_env(task, seed, training_num, test_num, obs_norm):
#     """Wrapper function for Webots env.
#
#     If EnvPool is installed, it will automatically switch to EnvPool's Webots env.
#
#     :return: a tuple of (single env, training envs, test envs).
#     """
#     if envpool is not None:
#         train_envs = env = envpool.make_gymnasium(
#             task, num_envs=training_num, seed=seed
#         )
#         test_envs = envpool.make_gymnasium(task, num_envs=test_num, seed=seed)
#     else:
#         warnings.warn(
#             "Recommend using envpool (pip install envpool) "
#             "to run Webots environments more efficiently."
#         )
#         print("test1")
#         env = gym.make(task)
#         # print("test2")
#         train_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(training_num)])
#         # print("test3")
#         test_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(test_num)])
#         train_envs.seed(seed)
#         test_envs.seed(seed)
#     if obs_norm:
#         # obs norm wrapper
#         train_envs = VectorEnvNormObs(train_envs)
#         test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
#         test_envs.set_obs_rms(train_envs.get_obs_rms())
#     return env, train_envs, test_envs
