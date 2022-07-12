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


        is_healthy = (self.state_vector()[2]) >-0.05 and (np.abs(self.state_vector()[3:6]) < 1.2).all()
        Map_array = [ "curved_map1" , "curved_map2" , "curved_map3", "curved_map4"]

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

        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]
            if(str(self.sim.model.geom_id2name(sim_contact.geom1)) == "Goal"):
                is_healthy = False
                print("Goal IN!")
                return is_healthy

            elif(str(self.sim.model.geom_id2name(sim_contact.geom2)) == "Goal"):
                is_healthy = False
                print("Goal IN!")
                return is_healthy

        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy
        return done

    def step(self, action):
        x_init = self.state_vector()[0]
        y_init = self.state_vector()[1]
        self.do_simulation(action, self.frame_skip)

        # update action and observation history
        
        # actually act_buffer will not be obs
        # self._act_buffer3 = self._act_buffer2
        # self._act_buffer2 = self._act_buffer1
        # self._act_buffer1 = action[:]
        
        
        self._obs_buffer3 = self._obs_buffer2
        self._obs_buffer2 = self._obs_buffer1
        self._obs_buffer1 = self.state_vector()[6:24]

        # calculate rewards and costs
        
        # goal_pos = np.array([15.0, 0.0])
        # pos_err = np.sqrt(np.mean(np.square(goal_pos - self.state_vector()[0:2])))

        x_vel = (self.state_vector()[0] - x_init)/self.dt
        y_vel = (self.state_vector()[1] - y_init)/self.dt


        xy_vel = np.sqrt(np.mean(np.square(np.array([x_vel,y_vel]))))


        if (xy_vel < 0.002 or x_vel < -0.1):
            reward = -5

        else :
            reward = 10*x_vel + abs(y_vel)

        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]

            if (str(self.sim.model.geom_id2name(sim_contact.geom1)) == "curved_map1"):
                reward -= 10
                print("Reward -= 10 : collision!")

            elif(str(self.sim.model.geom_id2name(sim_contact.geom2)) == "curved_map1"):
                reward -= 10
                print("Reward -= 10 : collision!")

            elif (str(self.sim.model.geom_id2name(sim_contact.geom1)) == "curved_map2" ):
                reward -= 10
                print("Reward -= 10 : collision!")

            elif(str(self.sim.model.geom_id2name(sim_contact.geom2)) == "curved_map2" ):
                reward -= 10
                print("Reward -= 10 : collision!")


            elif (str(self.sim.model.geom_id2name(sim_contact.geom1)) == "curved_map3" ):
                reward -= 10
                print("Reward -= 10 : collision!")

            elif(str(self.sim.model.geom_id2name(sim_contact.geom2)) == "curved_map3" ):
                reward -= 10
                print("Reward -= 10 : collision!")


            elif (str(self.sim.model.geom_id2name(sim_contact.geom1)) == "curved_map4" ):
                reward -= 10
                print("Reward -= 10 : collision!")

            elif(str(self.sim.model.geom_id2name(sim_contact.geom2)) == "curved_map4" ):
                reward -= 10
                print("Reward -= 10 : collision!")

        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]

            if(str(self.sim.model.geom_id2name(sim_contact.geom1)) == "Goal"):
                print("Goal IN!")
                reward = 2000

            elif(str(self.sim.model.geom_id2name(sim_contact.geom2)) == "Goal"):
                print("Goal IN!")
                reward = 2000

        # print("reward : ", reward)

        #y_err = np.abs(self.state_vector()[1])
        #ctrl = np.sum(np.square(self._act_buffer1 - self._act_buffer2))
        #torque_rms = np.sqrt(np.mean(np.square(self.sim.data.actuator_force[:])))
        #reward = x_del / (torque_rms + 1) / (y_err + 0.1)
        #reward = np.exp(-0.05 * (pos_err) ** (2))

        done = self.done
        observation = self._get_obs()
        info = {
               
            'total reward' : reward
        }

        return observation, reward, done, info

    def _get_obs(self):
        # take account of history
        #camera_data = np.array(self.render("rgb_array", 148, 148, 2))
        #data = self._get_viewer("rgb_array").read_pixels(100, 100, depth=False)

        data = self._get_viewer("rgb_array").read_pixels(148, 148, depth=False)
        CHW = np.transpose(data[::-1,:], (2, 0, 1))

        #print(CHW)

        #print(CHW.shape)
        ## if you wanna check the input image
        #plt.imshow(camera_data)
        #plt.show()

        #CHW = np.transpose(camera_data, (2, 0, 1))
        # flat_data = camera_data.flatten()
        # print(CHW.shape)

        obs_dct = {}

        obs_dct['image'] = CHW/255
        obs_dct['vector'] = self.state_vector()[6:24]

        #return np.concatenate([self._obs_buffer1,self._obs_buffer2,self._obs_buffer3])
        return obs_dct

    def reset_model(self):
        qpos = np.array([0,0.0,-0.005,0,0,0,0,-0.8,0.6,0,-0.8,0.6,0,-0.8,0.6,0,-0.8,0.6,0,-0.8,0.6,0,-0.8,0.6])
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


