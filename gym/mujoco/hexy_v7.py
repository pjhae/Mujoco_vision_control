import numpy as np
import matplotlib.pyplot as plt
from gym import utils
from gym.envs.mujoco import mujoco_env


# if you want to receive pixel data from camera using render("rgb_array",_,_,_,_)
# you should change below line <site_packages>/gym/envs/mujoco/mujoco_env.py to:
# self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, None, -1)


DEFAULT_CAMERA_CONFIG = {
    'distance': 1.5,
}

class HexyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_file='Hexy_ver_2.3/hexy-v2.3.xml',):
        utils.EzPickle.__init__(**locals())
        self._obs_buffer1 = np.zeros(18)
        self._heading_state = np.zeros(2)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def is_healthy(self):
        # if hexy was tilted or changed position too much, reset environments
        # is_healthy = np.abs(self.state_vector()[1]) < 0.5  and (np.abs(self.state_vector()[3:6]) < 0.7).all()

        is_healthy = ((self.state_vector()[2]) > -0.05 and (self.state_vector()[0]) < 0.15 and (self.state_vector()[1]) < 0.15)

        Map_array = [ "Map_circle1" , "Map_circle2" , "Map_circle3", "Map_circle4"]

        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]
            for j in range(len(Map_array)):
                if (str(self.sim.model.geom_id2name(sim_contact.geom1)) == Map_array[j]):
                    is_healthy = False
                    print("RESET : collision!")
                    return is_healthy
                elif (str(self.sim.model.geom_id2name(sim_contact.geom2)) == Map_array[j]):
                    is_healthy = False
                    print("RESET : collision!")
                    return is_healthy

        return is_healthy


    @property
    def done(self):
        done = not self.is_healthy
        return done

    def step(self, action):
    
        yaw_init = self.state_vector()[5]

        self.do_simulation(action, self.frame_skip)

        # update action and observation

        self._obs_buffer1 = self.state_vector()[6:24]

        # calculate rewards and costs
        
        z_ang_vel = (self.state_vector()[5] - yaw_init)
        torque_rms = np.sqrt(np.mean(np.square(self.sim.data.actuator_force[:])))

        # print(self.state_vector()[5] % (2*np.pi))

        if abs(z_ang_vel) < 0.00001:
        	reward = -1
        	
        else:
            if (self.state_vector()[5] % (2* np.pi) < np.pi):
                reward = (-10)*z_ang_vel/(torque_rms+1)+0.001

            else :
                reward = 10*z_ang_vel/(torque_rms+1)+0.001


        Map_array = [ "Map_circle1" , "Map_circle2" , "Map_circle3", "Map_circle4"]

        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]
            for j in range(len(Map_array)):
                if (str(self.sim.model.geom_id2name(sim_contact.geom1)) == Map_array[j]):
                    reward += -10
                    break

                elif (str(self.sim.model.geom_id2name(sim_contact.geom2)) == Map_array[j]):
                    reward += -10
                    break

        if (self.state_vector()[5] % (2 * np.pi) < np.pi / 30) :
            #print("GOAL IN [1] !!!!!")
            reward = 100

        elif (self.state_vector()[5] % (2 * np.pi) > 2 * np.pi - np.pi / 30):
            #print("GOAL IN [2] !!!!!")
            reward = 100


        done = self.done
        observation = self._get_obs()
        info = {
               
            'total reward' : reward
        }

        return observation, reward, done, info

    def _get_obs(self):
        self._obs_buffer1 = self.state_vector()[6:24]

        if (self.state_vector()[5] % (2 * np.pi) < np.pi):
            self._heading_state = [1, 0]

        else:
            self._heading_state = [0, 1]

        return np.append(self._obs_buffer1, self._heading_state)



    def reset_model(self):
        disc_factor = 2
        rnd = np.random.randint(disc_factor)
        
        init_yaw =  np.pi/disc_factor +  2*np.pi/disc_factor * rnd
        qpos = np.array([0.0,0.0,-0.005,0,0,init_yaw,0,-0.8,0.6,0,-0.8,0.6,0,-0.8,0.6,0,-0.8,0.6,0,-0.8,0.6,0,-0.8,0.6])
        
        qvel = self.init_qvel
        
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
