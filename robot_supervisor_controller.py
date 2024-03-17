import os
import time
from collections import deque

import gymnasium
from deepbots.supervisor import RobotSupervisorEnv
from gymnasium.spaces import Box, Dict, Tuple
import numpy as np
from controller import Lidar, Accelerometer, Robot, Supervisor
from utilities import normalize_to_range
from warnings import warn
from vehicle import Car
from scipy.interpolate import interp1d
from gymnasium.envs.registration import register
from gymnasium.utils.env_checker import check_env

import subprocess
import socket


class WebotsLauncher:
    def __init__(self, base_port=10000, max_port=11000):
        self.base_port = base_port
        self.max_port = max_port
        self.current_port = base_port

    def is_port_available(self, port):
        """Check if the specified port is available."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(('', port))
                sock.close()
                return True
            except socket.error:
                return False

    def find_available_port(self):
        """Find the next available port within the specified range."""
        for port in range(self.current_port, self.max_port + 1):
            if self.is_port_available(port):
                self.current_port = port + 1  # Prepare for the next search
                return port
        raise RuntimeError("No available ports found within the specified range.")

    def launch_webots(self, world_file):
        port = self.find_available_port()
        sim_proc = subprocess.Popen(["webots",
                                     "--minimize",
                                     "--batch",
                                     f'--port={port}',
                                     world_file])
        os.environ["WEBOTS_PID"] = str(sim_proc.pid)

        time.sleep(1)  # Wait for Webots to start
        for folder in os.listdir('/tmp'):
            if folder.startswith(f'webots-{sim_proc.pid}-'):
                try:
                    os.remove(F'/tmp/webots-{sim_proc.pid}')
                except FileNotFoundError:
                    pass
                os.symlink(F'/tmp/{folder}', F'/tmp/webots-{sim_proc.pid}')


# environnement
class RobotSupervisorController(RobotSupervisorEnv):
    """
       RobotSupervisorController acts as an environment having all the appropriate methods such as get_reward().
       This class utilizes the robot-supervisor scheme combining both the robot controls and the environment
       in the same class. Moreover, the reset procedure used is the default implemented reset.
       This class is made with the new release of deepbots in mind that fully integrates gymnasium.Env, using gymnasium.spaces.

        For this example, a simple custom 4 wheels(2 directional 4 motorized by one motor) driving robot is used, equipped with
        LiDAR (angle, distance), accelerometer (vector position, vector speed, vector acceleration) and gyrometer(vector orientation) sensors. The goal is to navigate through a track by modifying
        the speeds of the motor applied on each wheel, while avoiding obstacles using the LiDAR sensor.
        The agent observes its angle and distance values from LiDAR sensor, as well as its motor speeds, motor accelerations, orientations, and its latest action.
        The observation can also be augmented with observations from earlier steps.

        Observation:
        Type: Box(9)
        Num	Observation                     Min         Max
        0	Car Abs Position x (m)          -Inf        Inf       or Car Position Box(2,)   normalized to [-1, 1]
        1   Car Abs Position y (m)          -Inf        Inf
        2	Car Abs Velocity x (m/s)        -Inf        Inf        or Car Velocity Box(2,)   normalized to [-1, 1]
        3   Car Abs Velocity y (m/s)        -Inf        Inf
        4   Car Acceleration x (m/s²)       -Inf        Inf       or Car Acceleration Box(2,)    normalized to [-1, 1]
        5   Car Acceleration y (m/s²)       -Inf        Inf
        6   Car Abs Orientation x (deg)      0          360        or Car Orientation Box(2,)   normalized to [-1, 1]
        7   Car Abs Orientation y (deg)      0          360
        8	LiDAR points (m, deg)            0, 0     12, 360     or LiDAR Box(<scans>,)      normalized to [0, 1],[0, 1]


        Actions:
        Type: Continuous(2)
        Num	BodyPost      Min       Max
        0	Throttle    -1         1        normalized
        1	Direction   -1[-16]    1[+16]   normalized

        States:
        collision_state      bool        True if the vehicle collided with the wall
        pose                Array       Ground truth pose of the vehicle (x, y).
        acceleration        Array       Ground truth acceleration of the vehicle (x, y).
        Velocity            Array       Ground truth velocity of the vehicle (x, y).
        Lidar points        Array       Lidar points scanned describing obstacles (distance, angle)
        time                float       simu time (non implemented)
        checkpoint          int         Tracks are subdivided into checkpoints to make sure agents are racing in clockwise direction. Starts at index 3.
        wrong_way           bool        Indicates wether the agent goes in the right or wrong direction (not implemented)
        past_observations   deque       The n most recent observations of the agent in the past.
        past_actions	    deque	The n most recent actions of the agent in the past

        Reward:
         - Reward if the agent is close to the waypoint and changes its actions smoothly
         else penalize if the agent is far from the waypoint or changes its actions abruptly.
         - Reward the robot's distance parcoured from the previous position in one step.
         - Reward for reaching the target, i.e. decreasing the real distance under the threshold
         Final reward is modified by the time it took to reach the target
         - Negative -10 reward when crash.
         - Constant penalty of -0.0 at each time-step (to implement to test)

        Termination:
         Die : The car crashed
         Truncation : Episode length (timestep) is greater than 1000 (to precise)
         Solved Requirements : average episode score in last 100 episodes > 195.0 (to precise)
    """

    def __init__(self, render_mode=None, **kwargs):
        """
        In the constructor the observation_space and action_space are set and references to the various components
        of the robot required are initialized here.
        For the observation, we are using a time window of n_hist_obs to store the last n_hist_obs episode observations.
        """
        # self.car = kwargs["car"]
        # kwargs.pop("car")
        super().__init__(**kwargs)
        self.car = Car()
        self.car.setCruisingSpeed(0)
        # Lidar
        self.lidar = Lidar("RpLidarA2")
        self.lidar.enable(4*int(self.car.getBasicTimeStep()))
        # self.lidar.enablePointCloud()
        # Accelerometer
        self.accelerometer = Accelerometer("accelerometer")
        self.accelerometer.enable(4*int(self.car.getBasicTimeStep()))
        super(Supervisor, self).step(4*int(self.car.getBasicTimeStep()))

        # normalized value for position, velocity, acceleration, orientation and lidar_readings
        self.min_val, self.max_val = 0.0, 1.0

        # Apparition points
        dt = np.dtype([('position', np.float32, (3,)), ('rotation', np.float32, (4,))])
        self.appearance_points = np.array([
            ((-1.68, -2.68, 0.036), (0, 0, 1, 0)),
            ((1.77916, -2.90669, 0.036), (0, 0, 1, 0.785391)),
        ], dtype=dt)
        self.setup_agent()

        # Track waypoints
        self.track_number = 1
        self.n_waypoints = 4
        self.waypoints = self.initialize_waypoints(self.track_number)
        self.current_waypoint_index = 2

        self.previous_position = self.get_position()
        self.last_action = None

        # Car and sensors parameters
        engine_max_power_field = self.getSelf().getField("engineMaxPower")
        self.engineMaxPower = engine_max_power_field.getSFInt32() if engine_max_power_field else None
        max_steering_angle_field = self.getSelf().getField('maxSteeringAngle')
        # self.maxSteeringAngle = max_steering_angle_field.getSFFloat() if max_steering_angle_field else None
        self.maxSteeringAngle = 0.21
        min_steering_angle_field = self.getSelf().getField('minSteeringAngle')
        # self.minSteeringAngle = min_steering_angle_field.getSFFloat() if min_steering_angle_field else None
        self.minSteeringAngle = -0.21
        motor_velocity_field = self.getSelf().getField('maxVelocity')
        self.motor_velocity = motor_velocity_field.getSFFloat() if motor_velocity_field else None
        self.min_angle = 0  # lidar min wanted angle
        self.max_angle = 360  # lidar max wanted angle
        self.lidar_min_range = self.lidar.getMinRange()
        self.lidar_max_range = self.lidar.getMaxRange()
        self.N = 400  # reduced number of LiDAR points after SampleNet

        # Set up gymnasium spaces
        self.include_previous_observation = False
        self.n_hist_obs = 2  # number of observation history
        self.include_previous_action = False
        self.n_hist_act = 2  # number of action history

        # Buffers for past observations and actions
        self.past_observations = deque(maxlen=self.n_hist_obs)
        self.past_actions = deque(maxlen=self.n_hist_act)
        self.initialize_buffers()

        low_act = -self.max_val * np.ones(2, dtype=np.float32)
        max_act = self.max_val * np.ones(2, dtype=np.float32)
        self.action_space = Box(low=low_act, high=max_act, dtype=np.float32)

        # Base observation dimensions: pos(2,) + vel(2,) + acc(2,) + ori(2,) + lidar_points(N*2,)
        base_obs_dim = 2 + 2 + 2 + 2 + (self.N * 2)

        # Calculate total observation dimensions
        total_obs_dim = base_obs_dim
        if self.include_previous_observation:
            total_obs_dim += (base_obs_dim * self.n_hist_obs)  # Include previous observation
        if self.include_previous_action:
            total_obs_dim += (2 * self.n_hist_act)  # Assuming action space is 2D

        # Define bounds
        low_bounds = np.array([-self.max_val] * 8 + [self.min_val] * (self.N * 2), dtype=np.float32)
        high_bounds = np.array([self.max_val] * 8 + [self.max_val] * (self.N * 2), dtype=np.float32)

        if self.include_previous_observation:
            for _ in range(self.n_hist_obs):
                low_bounds = np.concatenate([low_bounds, low_bounds[:base_obs_dim]], dtype=np.float32)
                high_bounds = np.concatenate([high_bounds, high_bounds[:base_obs_dim]], dtype=np.float32)

        if self.include_previous_action:
            # Extend bounds to include previous action (assuming action space bounds [-1, 1])
            for _ in range(self.n_hist_act):
                low_bounds = np.concatenate([low_bounds, low_act], dtype=np.float32)  # Assuming each action is 2D
                high_bounds = np.concatenate([high_bounds, max_act], dtype=np.float32)

        # Create observation space
        self.observation_space = Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        self.r_dict = {}
        self.collision_threshold = 0.3  # 0.27
        self.collision_state = False

        self.max_score = 195
        self.episode_score_list = []
        self.terminate_on_collision = True
        self.time_limit = 1000

        self.state_gain = 0.1
        self.action_gain = 0.5
        self.on_target_threshold = 0.1
        self.collision_reward = -10

        # Logging
        self.max_distance_driven = 0.0
        self.distance_driven = 0.0

    def initialize_buffers(self):
        # Fill the past observations buffer with initial values
        for _ in range(self.n_hist_obs):
            initial_observation = np.concatenate([
                self.get_position(),
                self.get_velocity(),
                self.get_acceleration(),
                self.get_orientation(),
                self.get_lidar_readings(True)
            ], dtype=np.float32)
            self.past_observations.append(initial_observation)

        # Fill the past actions buffer with zeros
        for _ in range(self.n_hist_act):
            self.past_actions.append(np.zeros(2, dtype=np.float32))  # Assuming action space is 2D

    def get_default_observation(self):
        """
        This get_default_observation implementation builds the required observation for the AutonomousCar problem.
        A single observation consists of the car position, the car velocity (x,y axis), the car acceleration (x,y axis),
        the car orientation (x,y axis), the lidar points values.
        All values are normalized in their respective ranges, where appropriate:
        - Position are normalized to [-1.0, 1.0]
        - Velocities are normalized to [-1.0, 1.0]
        - Accelerations are normalized to [-1.0, 1.0]
        - Orientations are normalized to [-1.0, 1.0]
        - LiDAR points values are normalized to [0.0, 1.0]
        Optionally includes observation history and action history if specified.

        :return: Observation (potentially including previous observations and actions):
        [position, velocity, acceleration, orientation, lidar_points, prev_obs, prev_act]
        :rtype: nparray
        """

        position = self.get_position()
        velocity = self.get_velocity()
        acceleration = self.get_acceleration()
        orientation = self.get_orientation()
        lidar_readings = self.get_lidar_readings(True)

        # Concatenate current observation components
        core_observation = np.concatenate([position, velocity, acceleration, orientation, lidar_readings],
                                          dtype=np.float32)
        current_observation = core_observation

        # Include past observations if specified
        if self.include_previous_observation:
            for prev_obs in self.past_observations:
                current_observation = np.concatenate([current_observation, prev_obs], dtype=np.float32)

        # Include past actions if specified
        if self.include_previous_action:
            for prev_action in self.past_actions:
                current_observation = np.concatenate([current_observation, prev_action], dtype=np.float32)

        if self.include_previous_observation:
            self.past_observations.append(core_observation)
            while len(self.past_observations) > self.n_hist_obs:
                self.past_observations.popleft()
        assert self.observation_space.contains(current_observation)
        return current_observation

    def get_position(self):
        """
        Get the car's position from Webots.
        :return: current car position
        :rtype: nparray
        """
        position = self.getSelf().getPosition()
        return np.array(
            [normalize_to_range(p, -50.0, 50.0, -self.max_val, self.max_val, clip=True) for p in position[:2]],
            dtype=np.float32)  # Returning (x, y) components

    def get_velocity(self):
        """
        Get the car's velocity from Webots.
        :return: current car velocity
        :rtype: nparray
        """
        velocity = self.getSelf().getVelocity()
        return np.array([normalize_to_range(v, -self.motor_velocity, self.motor_velocity, -self.max_val,
                                            self.max_val, clip=True)
                         for v in velocity[:2]], dtype=np.float32)  # Returning (x, y) components of linear velocity

    def get_acceleration(self):
        """
        Get the car's acceleration from the accelerometer.
        :return: current car acceleration
        :rtype: nparray
        """
        acceleration_values = self.accelerometer.getValues()
        return np.array([normalize_to_range(a, -30.0, 30.0, -self.max_val, self.max_val, clip=True)
                         for a in acceleration_values[:2]], dtype=np.float32)  # Returning (x, y) components

    def get_orientation(self):
        """
        Get the car's orientation.
        :return: current car orientation
        :rtype: nparray
        """
        orientation = self.getSelf().getOrientation()
        return np.array([normalize_to_range(o, -np.pi, np.pi, -self.max_val, self.max_val, clip=True)
                         for o in orientation[:2]], dtype=np.float32)  # Returning (x, y) components

    def get_lidar_readings(self, norm=False):
        """
        Create a 2D numpy array with [distance, angle] for each LIDAR point within the specified range
        :return: current liaar scan of shape (N, 2) where each row is [distance, angle]
        :rtype: nparray
        """
        range_image = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        min_angle_rad = np.radians(self.min_angle - 180)  # Shift by 180° to make 0° the center
        max_angle_rad = np.radians(self.max_angle - 180)

        # Calculate angles for each LiDAR point
        # Assuming the LiDAR has a field of view from min_angle to max_angle
        angles = np.linspace(self.min_angle, self.max_angle, self.N, dtype=np.float32)

        # Convert angles from degrees to radians if necessary
        # angles_rad = np.deg2rad(angles)

        # Combine distances and angles
        lidar_readings = np.column_stack((range_image, angles))
        # Find indices where the LIDAR readings are 'inf'
        inf_indices = np.isinf(lidar_readings[:, 0])
        valid_indices = ~inf_indices  # Inverse of inf_indices: where LIDAR readings are valid
        # Interpolate to estimate 'inf' values
        if np.any(inf_indices):
            # Iterate over 'inf' indices and replace with the nearest valid value
            for index in np.where(inf_indices)[0]:
                # Find the nearest non-'inf' value on both sides
                left_nearest = np.max(range_image[:index][~np.isinf(range_image[:index])], initial=np.inf)
                right_nearest = np.min(range_image[index:][~np.isinf(range_image[index:])], initial=np.inf)

                # Choose the closest non-'inf' value to replace the 'inf'
                if left_nearest == np.inf and right_nearest == np.inf:
                    # In case all values are inf, this prevents an error, adjust as needed
                    continue
                elif left_nearest == np.inf:
                    lidar_readings[index, 0] = right_nearest
                elif right_nearest == np.inf:
                    lidar_readings[index, 0] = left_nearest
                else:
                    # If both sides have valid values, choose the nearest
                    lidar_readings[index, 0] = left_nearest if abs(
                        index - np.where(range_image == left_nearest)[0][0]) < abs(
                        index - np.where(range_image == right_nearest)[0][0]) else right_nearest
        if norm:
            # Normalize LiDAR readings
            lidar_distances = lidar_readings[:, 0]
            lidar_angles = lidar_readings[:, 1]
            lidar_distances_norm = np.array(
                [normalize_to_range(d, self.lidar_min_range, self.lidar_max_range, 0, 1, clip=True)
                 for d in lidar_distances], dtype=np.float32)
            lidar_angles_norm = np.array([normalize_to_range(a, self.min_angle, self.max_angle, 0, 1, clip=True)
                                          for a in lidar_angles], dtype=np.float32)
            lidar_readings = np.column_stack((lidar_distances_norm, lidar_angles_norm)).flatten()
        return lidar_readings

    def update_distance_driven(self):
        """
        Update the total distance driven by the car.
        """
        current_position = self.get_position()  # Get current position
        if self.previous_position is not None:
            # Calculate distance between current position and previous position
            distance = np.linalg.norm(current_position - self.previous_position)
            self.distance_driven += distance  # Update total distance driven

        # Update previous position for the next call
        self.previous_position = current_position

    def setup_agent(self):
        """
        This method initializes the position of the robot, storing the references inside a list and setting the starting
        positions and velocities.
        """
        trans_field = self.getSelf().getField("translation")
        rot_field = self.getSelf().getField("rotation")
        position, rotation = self.np_random.choice(self.appearance_points)
        trans_field.setSFVec3f(np.array(position).tolist())
        rot_field.setSFRotation(np.array(rotation).tolist())
        self.getSelf().resetPhysics()

    def initialize_waypoints(self, track_number):
        """
        Initialize the positions, size and orientation of the waypoints.
        """
        waypoints = []
        for i in range(1, self.n_waypoints + 1):
            waypoint_name = f'WAYPOINT_{i}({track_number})'
            waypoint_node = self.getFromDef(waypoint_name)  # Get the node using Webots API
            if waypoint_node is not None:

                translation_field = waypoint_node.getField('translation')
                position = translation_field.getSFVec3f() if translation_field else [0, 0, 0]

                bounding_object_field = waypoint_node.getField('boundingObject')
                if bounding_object_field:
                    box_node = bounding_object_field.getSFNode()
                    if box_node:
                        size_field = box_node.getField('size')
                        size = size_field.getSFVec3f()[:2] if size_field else [0, 0]
                    else:
                        size = [0, 0]
                else:
                    size = [0, 0]

                rotation_field = waypoint_node.getField('rotation')
                orientation = rotation_field.getSFRotation() if rotation_field else [0, 0, 1, 0]

                waypoints.append({
                    'position': position,
                    'size': size,
                    'orientation': orientation
                })
        return waypoints

    def get_next_waypoint(self):
        """
        Get the position of the next waypoint.
        """
        if self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]

    def update_waypoint_target(self):
        """
        Update the target waypoint index after reaching a waypoint.
        Loops back to the first waypoint after the last one is reached.
        """
        if self.current_waypoint_index < len(self.waypoints) - 1:
            self.current_waypoint_index += 1
        else:
            # Loop back to the first waypoint after the last one is reached
            self.current_waypoint_index = 0

    def get_closest_point_on_line(self, agent_position, cur_waypoint):
        """
        Calculate the closest point on the waypoint line to the agent's current position.

        Args:
            agent_position: The current position of the agent.
            cur_waypoint: A dictionary containing the 'position', 'size', and 'orientation' of the current waypoint.

        Returns:
            The closest point on the waypoint line to the agent_position.
        """
        line_segment = self.get_line_segment(cur_waypoint)
        line_start, line_end = line_segment[0], line_segment[1]

        # Calculate the closest point on the infinite line
        line_vec = line_end - line_start
        agent_vec = agent_position[:2] - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        agent_proj = np.dot(agent_vec, line_unitvec)
        closest_point = np.clip(agent_proj, 0, line_len) * line_unitvec + line_start

        return closest_point

    def get_line_segment(self, waypoint):
        """
        Calculate the start and end points of the waypoint line segment based on its position, size, and orientation.

        Args:
            waypoint: A dictionary containing the 'position', 'size', and 'orientation' of the waypoint line.

        Returns:
            A tuple (line_start, line_end) representing the start and end points of the line segment.
        """

        position = np.array(waypoint['position'])
        size = np.array(waypoint['size'])
        orientation = np.array(waypoint['orientation'])

        # Since the rotation is about the z-axis, we can ignore the axis part and use only the angle.
        angle = orientation[3]  # This is the angle of rotation about the z-axis.

        # Calculate the direction vector in the x-y plane after rotation by the given angle.
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)

        # The direction vector for the x-axis rotated by 'angle' radians.
        direction = np.array([cos_angle, sin_angle])

        half_size = size[0] / 2.0  # Assuming the length of the plane is along the x-axis after rotation
        line_start = position[:2] - direction * half_size  # Only use x and y for 2D calculations
        line_end = position[:2] + direction * half_size

        return np.array([line_start, line_end])

    def get_reward(self, action):
        reward = 0.0

        # Reward if the agent is close to the waypoint and changes its actions smoothly
        # else penalize if the agent is far from the waypoint or changes its actions abruptly
        position = self.get_position()
        waypoint = self.get_next_waypoint()

        # Find the closest point on the waypoint line to the agent's current position
        closest_point_on_waypoint = self.get_closest_point_on_line(position, waypoint)
        # Calculate distance to the closest point on the waypoint line
        distance_to_waypoint = np.linalg.norm(position - closest_point_on_waypoint)

        Q = self.state_gain * np.identity(len(position))
        R = self.action_gain * np.identity(len(action))
        delta_pos = closest_point_on_waypoint - position
        delta_act = np.array(action) - np.array(self.last_action)
        cost = (np.matmul(delta_pos, np.matmul(Q, delta_pos)) + np.matmul(delta_act, np.matmul(R, delta_act)))
        reward += np.exp(-cost)
        self.r_dict['Close from W + smooth rew'] = reward

        # Reward for reaching the target, i.e. decreasing the real distance under the threshold
        # Final reward is modified by the time it took to reach the target
        reach_tar_reward = 0.0
        if distance_to_waypoint < self.on_target_threshold:
            reach_tar_reward = 1.0 - 0.5 * self.timestep / self.time_limit  # timestep in ms
        reward += reach_tar_reward
        self.r_dict['Reach W rew'] = reward

        # Reward higher speeds but up to a certain limit
        # to prevent the car from recklessly maximizing speed without regard to safety/track conditions
        dist_moved = np.linalg.norm([position[0] - self.previous_position[0],
                                     position[1] - self.previous_position[1]])
        speed_reward = normalize_to_range(dist_moved, 0.0, 8.0, -1.0, 1.0)
        reward += speed_reward
        self.r_dict['Dist moved rew'] = reward

        # Big penalty if collision.
        if self.check_collision():
            reward += self.collision_reward
            self.collision_state = True
        self.r_dict['Collision rew'] = reward
        print(self.r_dict)
        self.episode_score_list.append(reward)
        return reward

    def is_terminated(self):
        """
        This method checks the termination criteria for each episode.
        If the criteria are satisfied it returns True otherwise it returns False.
        Criteria :
            Die : The car crashed, or is off the track
            Solved Requirements : average episode score in last 100 episodes > 195.0 (to precise)
        :return: Whether the termination criteria have been met.
        :rtype: bool
        """
        if self.terminate_on_collision and self.collision_state:
            print("Collison !!!")
            return True
        # Check the z-position termination condition.
        position = self.getSelf().getPosition()
        if position[2] < -0.5 or position[2] > 0.5:
            print("Sortie de piste !!!")
            return True
        # Check the rotation termination condition.
        rot_field = self.getSelf().getField("rotation")
        rotation = rot_field.getSFRotation()
        if 2 <= rotation[3] <= 4:
            print("Agent retourné !!!")
            return True
        if len(self.episode_score_list) >= 100:
            average_score = np.mean(self.episode_score_list[-100:])
        else:
            average_score = np.mean(self.episode_score_list)
        return average_score > self.max_score

    def is_truncated(self):
        """
        This method checks the truncation criteria for each episode.
        If the criteria are satisfied it returns True otherwise it returns False.
        Criteria :
            Truncation : Episode length (timestep) is greater than 1000 (to precise)
        :return: Whether the truncation criteria have been met.
        :rtype: bool
        """
        print("is_truncated")
        return self.time_limit < self.timestep

    def reset(self, seed=None, options=None):
        """
        This method overrides reset in SupervisorEnv to reset a few variables.
        :return: observation provided by the following get_default_observation()
        """
        # super().reset(seed=seed)
        # self.simulationReset()
        self.simulationResetPhysics()
        self.r_dict = {}
        self.current_waypoint_index = 2
        self.collision_state = False
        self.setup_agent()
        self.previous_position = self.get_position()
        self.distance_driven = 0.0
        self.waypoints = self.initialize_waypoints(track_number=self.track_number)
        #super(Supervisor, self).step(4*int(self.car.getBasicTimeStep()))
        print("reset")
        self.past_observations = deque(maxlen=self.n_hist_obs)
        self.past_actions = deque(maxlen=self.n_hist_act)
        self.initialize_buffers()
        return self.get_default_observation(), self.get_info()

    def apply_action(self, action):
        """
        This method uses the action list provided, which contains the next action to be
        executed as float numbers denoting the action of the car.
        The corresponding throttle value and steering are applied one the car.
        Args:
            action: A list or array with two elements:
                action[0] is the throttle (range: -1 to 1, where -1 is full reverse and 1 is full forward)
                action[1] is the direction (range: -1 to 1, corresponding to minSteeringAngle to maxSteeringAngle)
        :param action: The list that contains the action values
        :type action: list of floats
        """
        throttle_action = action[0]
        steering_action = action[1]
        # Map the normalized action values to actual control values
        # throttle = throttle_action * self.engineMaxPower  # Scale throttle to engine's power range
        #throttle = np.abs(throttle_action * self.motor_velocity / 10)
        throttle = ((throttle_action + 1) / 2) * self.motor_velocity/2
        steering_angle = steering_action * (self.maxSteeringAngle - self.minSteeringAngle) / 2 + \
                         (self.maxSteeringAngle + self.minSteeringAngle) / 2  # Map direction to steering angle range
        if throttle_action < 0:
            self.car.setGear(-1)  # Reverse gear
        else:
            self.car.setGear(1)  # Forward gears

        # Apply throttle and steering
        self.car.setBrakeIntensity(0.0)
        # self.car.setThrottle(throttle)
        self.car.setCruisingSpeed(throttle)
        self.car.setSteeringAngle(steering_angle)
        print(f"Action: {action}, Throttle: {throttle}, Steering angle: {steering_angle}")

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action):
        """
        Step override method which slightly modifies the parent.
        It applies the previous action, steps the simulation, updates the metrics with new values and then
        gets the new observation, reward, done flag and info and returns them.

        :param action: The action to perform
        :return: new observation, reward, done flag, info
        :rtype: tuple
        """
        self.apply_action(action)

        # Update the action history with the latest action if include_actions is True
        if self.include_previous_action:
            # Ensure the action is a NumPy array and append it to the action history
            self.past_actions.append(np.array(action, dtype=np.float32))
            # Ensure that only the last n_hist_obs observations are retained
            while len(self.past_actions) > self.n_hist_act:
                self.past_actions.popleft()

        if super(Supervisor, self).step(4*int(self.car.getBasicTimeStep())) == -1:  # NOQA
            exit()
        self.last_action = np.array(action, dtype=np.float32)
        return (
            self.get_default_observation(),
            self.get_reward(action),
            self.is_terminated(),
            False,
            self.get_info()
        )

    def get_info(self):
        # self.update_distance_driven()
        # if self.distance_driven > self.max_distance_driven:
        #     self.max_distance_driven = self.distance_driven
        # position = self.get_position()
        # waypoint = self.get_next_waypoint()
        # closest_point_on_waypoint = self.get_closest_point_on_line(position, waypoint)
        # return {
        #     "curr_distance": self.distance_driven,
        #     "max_distance": self.max_distance_driven,
        #     "dist2Waypoint": np.linalg.norm(position - closest_point_on_waypoint)
        # }
        return {}

    def render(self, mode='human'):
        print("not rendering")

    def check_collision(self):
        """
        Check for potential collisions based on LiDAR readings within a 45° cone in front of the robot.
        Returns True if a potential collision is detected, otherwise False.
        """
        lidar_readings = self.get_lidar_readings()  # Assuming this returns an array of [distance, angle] pairs
        # print(lidar_readings)
        half_cone_angle = 96 / 2  # Half the cone angle

        # Filter the readings to only those within the 45° cone in front of the robot
        readings_in_cone = lidar_readings[(lidar_readings[:, 1] >= -half_cone_angle) &
                                          (lidar_readings[:, 1] <= half_cone_angle)]

        # Check if any distances in the cone are below the collision threshold
        if np.any(readings_in_cone[:, 0] < self.collision_threshold):
            return True  # Collision detected

        return False  # No collision detected


if __name__ == '__main__':
    car = Car()
    kwargs = dict()
    kwargs["car"] = car
    controller = RobotSupervisorController(**kwargs)
    # register(
    #     id="VektorE-v0",
    #     entry_point="robot_supervisor_controller_Box:RobotSupervisorController"
    # )
    # env = gymnasium.make("VektorE-v0", **kwargs)
#     obs = env.observation_space
#     act = env.action_space
#     state_shape = env.observation_space.shape or env.observation_space.n
#     action_shape = env.action_space.shape or env.action_space.n
#     max_action = env.action_space.high[0]
#     print("Observation space: ", obs)
#     print("Observations shape:", state_shape)
#     print("Observation range: ", np.min(env.observation_space.low), np.max(env.observation_space.high))
#     print("Action space: ", act)
#     print("Actions shape:", action_shape)
#     print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
#     print("Reset", env.reset()[0].tolist())
#     print("Result step function: ", env.step([0.0, 0.0])[0])
#     check_env(env.unwrapped)

# print(controller.reset()[0])
#     print(controller.car.getType())
#     print(controller.car.getEngineType())
    while super(Supervisor, controller).step(4) != -1:
        # """ Apply action """
        controller.apply_action([1.0, 1.0])
# print(controller.car.getControlMode())

# """ Accelerometer """
# print(controller.accelerometer.getValues())

# """ LiDAR """
# lidar_r = controller.get_lidar_readings()
# print(lidar_r)

# """ Observation """
# print("Default obs: ",controller.get_default_observation())
# controller.current_observation = controller.get_default_observation()
# print("Current obs: ", controller.current_observation)
# obs = controller.get_default_observation()
# print("Send obs: ", obs)
# controller.observation_history.append(controller.current_observation)
# # Ensure that only the last n_hist_obs observations are retained
# while len(controller.observation_history) > controller.n_hist_obs:
#     controller.observation_history.popleft()

# """ Collision check """
# collision = controller.check_collision()
# print("Collision ?: ", collision)

# """ Rewards : play with last_action/action and the car position in relation to current waypoint pos """
# controller.last_action = [0.3, -0.1]
# rewards = controller.get_reward([0.4, -0.2])
# print("Final rew: ", rewards)

# """ Find the closest point on the waypoint line to the agent's current position """
# position = controller.get_position()
# waypoint = controller.get_next_waypoint()
# closest_point_on_waypoint = controller.get_closest_point_on_line(position, waypoint)
# distance_to_waypoint = np.linalg.norm(position - closest_point_on_waypoint)
# print("Waypoint", waypoint.values(), "\ndistance :", distance_to_waypoint)

# """ Apply action """
# print(controller.car.getControlMode())
# controller.apply_action([0.8, 0.2])

# controller.car.setCruisingSpeed(0.1)
# angle = 0.3*np.cos(controller.getTime())
# controller.car.setSteeringAngle(angle)
