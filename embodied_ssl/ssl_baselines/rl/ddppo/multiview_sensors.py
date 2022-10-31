from typing import Any

import numpy as np
import quaternion
import math
from gym import spaces
from torchvision import transforms

import os
import uuid

from habitat import registry, Simulator, Config
from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator


@registry.register_sensor
class RGB360Sensor(Sensor):
    cls_uuid = "rgb_360"
    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):

        self._sim = sim
        self._num_views = getattr(config, "NUM_VIEWS")

        self._data_aug = False
        if hasattr(config, "DATA_AUG"):
            self._data_aug = getattr(config, "DATA_AUG")
        
        if self._data_aug:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(
                    size = (128, 128),
                    scale = (0.8, 1.0),
                    ratio = (1.0, 1.0),
                ),
                transforms.ColorJitter(
                    brightness = 0.2,
                    contrast = 0.2,
                    saturation = 0.2,
                    hue = 0.2,
                ),
                transforms.ToTensor(),
            ])

        rotation_angle = math.pi/self._num_views
        self._cos = math.cos(rotation_angle)
        self._sin = math.sin(rotation_angle)
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        rgb_shape = self._sim.sensor_suite.observation_spaces.spaces['rgb'].shape
        return spaces.Box(
            low=0,
            high=255,
            shape=(self._num_views, 3, rgb_shape[0], rgb_shape[1]),
            dtype=np.uint8,
        )

    def transform_obs(self, obs):
        if self._data_aug:
            return 255 * self.transform(obs)
        else:
            return obs.transpose((2, 0, 1))

    # This is called whenver reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, **kwargs: Any
    ):
        observations_360 = [self.transform_obs(observations["rgb"])]

        position = self._sim.get_agent_state().position
        rotation = self._sim.get_agent_state().rotation
        rotation_y = np.quaternion(self._cos, 0, -self._sin, 0)

        for _ in range(self._num_views - 1):
            rotation *= rotation_y
            observations_y = self._sim.get_observations_at(position, rotation)
            transformed_obs = self.transform_obs(observations_y["rgb"])
            observations_360.append(transformed_obs)

        out = np.stack(observations_360, axis = 0)
        return out


@registry.register_sensor
class ImageGoal360Sensor(RGB360Sensor):
    r"""Sensor for ImageGoal observations which are used in ImageGoal Navigation.
    RGBSensor needs to be one of the Simulator sensors.
    This sensor return the rgb image taken from the goal position to reach with
    random rotation.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the ImageGoal sensor.
    """
    cls_uuid: str = "imagegoal_360"

    def _get_pointnav_episode_image_goal(self, episode: Episode):
        goal_position = np.array(episode.goals[0].position, dtype=np.float32)
        # to be sure that the rotation is the same for the same episode_id
        # since the task is currently using pointnav Dataset.
        seed = abs(hash(episode.episode_id)) % (2 ** 32)
        rng = np.random.RandomState(seed)
        angle = rng.uniform(0, 2 * np.pi)
        source_rotation = np.quaternion(np.sin(angle / 2), 0, np.cos(angle /
            2), 0)
        goal_observation = self._sim.get_observations_at(
            position=goal_position.tolist(), rotation=source_rotation
        )
        observations_360 = [self.transform_obs(goal_observation['rgb'])]
        rotation_y = np.quaternion(self._cos, 0, -self._sin, 0)

        for _ in range(self._num_views - 1):
            source_rotation *= rotation_y
            observations_y = self._sim.get_observations_at(goal_position,
                    source_rotation)
            transformed_obs = self.transform_obs(observations_y["rgb"])
            observations_360.append(transformed_obs)

        out = np.stack(observations_360, axis = 0)
        return out

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ):
        if not hasattr(self, "_current_episode_id"):
            self._current_episode_id = None

        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_image_goal

        self._current_image_goal = self._get_pointnav_episode_image_goal(
            episode
        )
        self._current_episode_id = episode_uniq_id

        return self._current_image_goal

