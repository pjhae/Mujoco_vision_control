import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


DEFAULT_CAMERA_CONFIG = {
    'distance': 1.5,
}


class HexyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, xml_file='Hexy_ver_2.3/hexy-v2.3.xml',):
        utils.EzPickle.__init__(**locals())
        self._obs_buffer1 = np.zeros(18)
        self._obs_buffer2 = np.zeros(18)
        self._obs_buffer3 = np.zeros(18)
        self._act_buffer1 = np.zeros(18)
        self._act_buffer2 = np.zeros(18)
        self._act_buffer3 = np.zeros(18)
        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property
    def is_healthy(self):
        # if hexy was tilted or changed position too much, reset environments
        # is_healthy = np.abs(self.state_vector()[1]) < 0.5  and (np.abs(self.state_vector()[3:6]) < 0.7).all()
        is_healthy = (self.state_vector()[2]) >-0.05 and (np.abs(self.state_vector()[3:6]) < 0.7).all()
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy
        return done

    def step(self, action):
        x_init = self.state_vector()[0]
        self.do_simulation(action, self.frame_skip)

        # update action and observation history
        
        # actually act_buffer will not be obs
        self._act_buffer3 = self._act_buffer2
        self._act_buffer2 = self._act_buffer1
        self._act_buffer1 = action[:]
        
        
        self._obs_buffer3 = self._obs_buffer2
        self._obs_buffer2 = self._obs_buffer1
        self._obs_buffer1 = self.state_vector()[6:24]

        # calculate rewards and costs
        
        #goal_pos = np.array([15,-0.5])
        
        #pos_err = np.sqrt(np.mean(np.square(goal_pos - self.state_vector()[0:2])))
        
        
        
        x_del = self.state_vector()[0] - x_init  #x_init
        y_err = np.abs(self.state_vector()[1])
        yaw = np.abs(self.state_vector()[1])
        ctrl = np.sum(np.square(self._act_buffer1 - self._act_buffer2))
        torque_rms = np.sqrt(np.mean(np.square(self.sim.data.actuator_force[:])))
        reward = x_del / (torque_rms + 1) / (y_err + 0.1)
        #reward = x_del

        # print("x:",self.state_vector()[0])
        # print("y:",self.state_vector()[1])
        # print("z:",self.state_vector()[2])
        # sim_contact = self.sim.data.contact
        # print(sim_contact.geom2)
        # print(str(self.sim.model.geom_id2name(sim_contact.geom2)))
        # print(self.state_vector()[1])
        
        
        #for i in range(self.sim.data.ncon):
        #    sim_contact = self.sim.data.contact[i]
        #    contact_body_name =str(self.sim.model.geom_id2name(sim_contact.geom1))+"+"+str(self.sim.model.geom_id2name(sim_contact.geom2))
        #    print(contact_body_name)
            
  
        
        done = self.done
        observation = self._get_obs()
        info = {
            'x_delta' : x_del,
            'y_error' : y_err,
            'control_norm' : ctrl,
            'total' : reward
        }

        return observation, reward, done, info

    def _get_obs(self):
        # take account of history
        return np.concatenate([self._obs_buffer1, self._obs_buffer2, self._obs_buffer3])
    def reset_model(self):
        qpos = np.array([0,0,-0.022,0,0,0,0,-0.5,0.5,0,-0.5,0.5,0,-0.5,0.5,0,-0.5,0.5,0,-0.5,0.5,0,-0.5,0.5])
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
