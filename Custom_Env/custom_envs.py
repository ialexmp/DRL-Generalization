import math
from typing import List, Optional
from venv import logger
import gym
from gym.envs.registration import register
import gymnasium_robotics as gym

from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv, MujocoPyFetchPickAndPlaceEnv
from gym.envs.classic_control import *
import numpy as np

""" 
 ## Rewards
  The reward can be initialized as `sparse` or `dense`:
  - *sparse*: the returned reward can have two values: `-1` if the block hasn't reached its final target position, and `0` if the block is in the final target position (the block is considered to have reached the goal if the Euclidean distance between both is lower than 0.05 m).
  - *dense*: the returned reward is the negative Euclidean distance between the achieved goal position and the desired goal.
 """

""" PICK AND PLACE CUSTOM ENVIRONMENTS """

class CustomFetchPickAndPlaceEnv(MujocoFetchPickAndPlaceEnv):
    def __init__(self, **kwargs):
        # Call the constructor of the parent class with modified parameters
        super().__init__(**kwargs)
        
        # Override parameters
        self.target_in_the_air = False
        self.reward_type = 'dense'
    
    def _reset_sim(self):
        # Call the parent class's _reset_sim() function
        super()._reset_sim()

        # Reset simulation time, joint positions, and velocities
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Set fixed start position for object
        if self.has_object:
            # Define fixed position for object
            object_x = 1.3419  # x-coordinate
            object_y = 0.7491  # y-coordinate
            object_z = 0.42     # z-coordinate (fixed height)
            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:3] = [object_x, object_y, object_z]
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        # Set fixed start position for target
        target_x = 1.0  # x-coordinate (different from object)
        target_y = 0.5  # y-coordinate (different from object)
        target_z = 0.42  # z-coordinate (fixed height)
        self.goal = [target_x, target_y, target_z]  # Define fixed position for target

        # Forward simulation
        self._mujoco.mj_forward(self.model, self.data)

        # Return True to indicate successful reset
        return True

class CustomPyFetchPickAndPlaceEnv(MujocoPyFetchPickAndPlaceEnv):
    def __init__(self, **kwargs):
        # Call the constructor of the parent class with modified parameters
        super().__init__(**kwargs)
        
        # Override parameters
        self.target_in_the_air = False
        
        # Additional modifications or initialization if needed

# Register CustomFetchPickAndPlaceEnv
gym.register(
    id='CustomFetchPickAndPlace',
    entry_point=__name__ + ':CustomFetchPickAndPlaceEnv',
)

# Register CustomPyFetchPickAndPlaceEnv
gym.register(
    id='CustomPyFetchPickAndPlace',
    entry_point=__name__ + ':CustomPyFetchPickAndPlaceEnv',
)


""" MOUNTAIN CAR CONTINUOUS CUSTOM ENVIRONMENTS """
class CustomMountainCarGoalPositionEnv(Continuous_MountainCarEnv):
    def __init__(self, goal_position=0.45, **kwargs):
        super().__init__(**kwargs)
        self.goal_position = goal_position 
                 
class CustomMountainCarPowerEnv(Continuous_MountainCarEnv):
    def __init__(self, power=0.0015, **kwargs):
        super().__init__(**kwargs)
        self.power = power # Custom power

class CustomMountainCarInitialPositionEnv(Continuous_MountainCarEnv):
    def __init__(self, initial_pos=-0.05,  **kwargs):
        super().__init__(**kwargs)
        self.initial_pos = initial_pos

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = np.array([self.initial_pos, 0], dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

class CustomMountainCarFrictionEnv(Continuous_MountainCarEnv):
    def __init__(self, car_friction= 0.0025, **kwargs):
        super().__init__(**kwargs)
        self.car_friction = car_friction

    def step(self, action: np.ndarray):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        # Increase friction effect
        velocity += force * self.power - self.car_friction * math.cos(3 * position)  # Modified friction
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        reward = 0
        if terminated:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1

        self.state = np.array([position, velocity], dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        return self.state, reward, terminated, False, {}

class CustomMountainCarRandomPerturbationsEnv(Continuous_MountainCarEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action: np.ndarray):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        noise = np.random.normal(0, 0.001)  # Introduce noise
        velocity += force * self.power - 0.0025 * math.cos(3 * position) + noise
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )

        reward = 0
        if terminated:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1

        self.state = np.array([position, velocity], dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        return self.state, reward, terminated, False, {}

register(
    id='CustomMountainCarRandomPerturbations',
    entry_point='__main__:CustomMountainCarRandomPerturbationsEnv',
)

register(
    id='CustomMountainCarFriction',
    entry_point='__main__:CustomMountainCarFrictionEnv',
)

register(
    id='CustomMountainCarInitialPosition',
    entry_point='__main__:CustomMountainCarInitialPositionEnv',
)

register(
    id='CustomMountainCarPower',
    entry_point='__main__:CustomMountainCarPowerEnv',
)


register(
    id='CustomMountainCarGoalPosition',
    entry_point='__main__:CustomMountainCarGoalPositionEnv',
)

""" CARTPOLE CUSTOM ENVIRONMENTS """

class CustomCartPoleLengthEnv(CartPoleEnv):
    def __init__(self, lenght = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.length = lenght # Custom pole length

class CustomCartPoleMassEnv(CartPoleEnv):
    def __init__(self, masspole = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.masspole = masspole  # Custom pole mass

class CustomCartPoleGravityEnv(CartPoleEnv):
    def __init__(self, gravity = 9.8 , **kwargs):
        super().__init__(**kwargs)
        self.gravity = gravity  # Custom gravity
        
class CustomCartPoleForceEnv(CartPoleEnv):
    def __init__(self, force_mag_left=10.0, force_mag_right=10.0, **kwargs):
        super().__init__(**kwargs)
        self.force_mag_left = force_mag_left
        self.force_mag_right = force_mag_right

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag_right if action == 1 else -self.force_mag_left
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

class CustomCartPoleRandomPerturbationsEnv(CartPoleEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, action):
        state, reward, done, info = super().step(action)
        perturbation = np.random.uniform(-0.01, 0.01, size=4)  # Introduce random perturbations
        state = np.add(state, perturbation)
        return state, reward, done, info

# Register CustomCartPoleLengthEnv
register(
    id='CustomCartPoleLength',
    entry_point=__name__ + ':CustomCartPoleLengthEnv',
)

# Register CustomCartPoleMassEnv
register(
    id='CustomCartPoleMass',
    entry_point=__name__ + ':CustomCartPoleMassEnv',
)

# Register CustomCartPoleGravityEnv
register(
    id='CustomCartPoleGravity',
    entry_point=__name__ + ':CustomCartPoleGravityEnv',
)

# Register CustomCartPoleForceEnv
register(
    id='CustomCartPoleForce',
    entry_point=__name__ + ':CustomCartPoleForceEnv',
)

# Register CustomCartPoleRandomPerturbationsEnv
register(
    id='CustomCartPoleRandomPerturbations',
    entry_point=__name__ + ':CustomCartPoleRandomPerturbationsEnv',
)

""" ACROBOT CUSTOM ENVIRONMENTS """

class ModifiedAcrobotLinkLengthsEnv(AcrobotEnv):
    def __init__(self, link_length_1, link_length_2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LINK_LENGTH_1 = link_length_1
        self.LINK_LENGTH_2 = link_length_2
        
class ModifiedAcrobotLinkMassesEnv(AcrobotEnv):
    def __init__(self, link_mass_1, link_mass_2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LINK_MASS_1 = link_mass_1
        self.LINK_MASS_2 = link_mass_2
        
# position of the center of mass        
class ModifiedAcrobotCoMPositionsEnv(AcrobotEnv):
    def __init__(self, com_pos_1, com_pos_2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LINK_COM_POS_1 = com_pos_1
        self.LINK_COM_POS_2 = com_pos_2
        
class ModifiedAcrobotMomentsOfInertiaEnv(AcrobotEnv):
    def __init__(self, moi_1, moi_2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LINK_MOI = moi_1  # Assuming moments of inertia are the same for both links

class ModifiedAcrobotMaxVelocitiesEnv(AcrobotEnv):
    def __init__(self, max_vel_1: float = 3 * math.pi, max_vel_2: float = 6 * math.pi, **kwargs):
        super().__init__(**kwargs)
        self.MAX_VEL_1 = max_vel_1
        self.MAX_VEL_2 = max_vel_2

class ModifiedAcrobotActionSpaceEnv(AcrobotEnv):
    def __init__(self, avail_torque: List[float] = [-1.0, 0.0, 1.0, 2.0], **kwargs):
        super().__init__(**kwargs)
        self.AVAIL_TORQUE = avail_torque

# Register ModifiedAcrobotLinkLengthsEnv
register(
    id='ModifiedAcrobotLinkLengths',
    entry_point=__name__ + ':ModifiedAcrobotLinkLengthsEnv',
)

# Register ModifiedAcrobotLinkMassesEnv
register(
    id='ModifiedAcrobotLinkMasses',
    entry_point=__name__ + ':ModifiedAcrobotLinkMassesEnv',
)

# Register ModifiedAcrobotCoMPositionsEnv
register(
    id='ModifiedAcrobotCoMPositions',
    entry_point=__name__ + ':ModifiedAcrobotCoMPositionsEnv',
)

# Register ModifiedAcrobotMomentsOfInertiaEnv
register(
    id='ModifiedAcrobotMomentsOfInertia',
    entry_point=__name__ + ':ModifiedAcrobotMomentsOfInertiaEnv',
)

# Register ModifiedAcrobotMaxVelocitiesEnv
register(
    id='ModifiedAcrobotMaxVelocities',
    entry_point=__name__ + ':ModifiedAcrobotMaxVelocitiesEnv',
)

# Register ModifiedAcrobotActionSpaceEnv
register(
    id='ModifiedAcrobotActionSpace',
    entry_point=__name__ + ':ModifiedAcrobotActionSpaceEnv',
)

""" PENDULUM CUSTOM ENVIRONMENTS """

def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi
    
class CustomPendulumLenghtEnv(PendulumEnv):
    def __init__(self, render_mode: Optional[str] = None, g=9.8, l=1.0):
        super().__init__(render_mode=render_mode, g=g)
        self.l = l  # Set the new length of the pendulum
        
    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l  # Use the custom length
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), -costs, False, False, {}


class CustomPendulumMassEnv(PendulumEnv):
    def __init__(self, m = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.m = m  # Custom Pendulum mass

class CustomPendulumDeltaTimeEnv(PendulumEnv):
    def __init__(self, dt = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt  # Custom Pendulum mass
        
class CustomPendulumGravityEnv(PendulumEnv):
    def __init__(self, g = 9.8 , **kwargs):
        super().__init__(**kwargs)
        self.g = g  # Custom gravity

# Register CustomPendulumLengthEnv
register(
    id='CustomPendulumLenght',
    entry_point=__name__ + ':CustomPendulumLenghtEnv',
)
# Register CustomPendulumLengthEnv
register(
    id='CustomPendulumMass',
    entry_point=__name__ + ':CustomPendulumMassEnv',
)
# Register CustomPendulumGravityEnv
register(
    id='CustomPendulumDeltaTime',
    entry_point=__name__ + ':CustomPendulumDeltaTimeEnv',
)
# Register CustomPendulumGravityEnv
register(
    id='CustomPendulumGravity',
    entry_point=__name__ + ':CustomPendulumGravityEnv',
)