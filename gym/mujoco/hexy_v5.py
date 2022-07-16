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
        is_healthy = (self.state_vector()[2]) >-0.05 and (np.abs(self.state_vector()[3:6]) < 1.3).all()

        Map_array = [ "curved_map1" , "curved_map2" , "curved_map3", "curved_map4"]

        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]
            for j in range(len(Map_array)):
                if (str(self.sim.model.geom_id2name(sim_contact.geom1)) == Map_array[j]):
                    is_healthy = False
                    print("Collision! : RESET")
                    return is_healthy
                elif (str(self.sim.model.geom_id2name(sim_contact.geom2)) == Map_array[j]):
                    is_healthy = False
                    print("Collision! : RESET")
                    return is_healthy

        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]
            if(str(self.sim.model.geom_id2name(sim_contact.geom1)) == "Goal"):
                is_healthy = False
                print("Goal IN! : RESET")
                return is_healthy

            elif(str(self.sim.model.geom_id2name(sim_contact.geom2)) == "Goal"):
                is_healthy = False
                print("Goal IN! : RESET")
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

        x_vel = (self.state_vector()[0] - x_init)/self.dt
        y_vel = (self.state_vector()[1] - y_init)/self.dt

        xy_vel = np.sqrt(np.mean(np.square(np.array([x_vel,y_vel]))))


        if (xy_vel < 0.002 or x_vel < -0.1):
            reward = -5

        else :
            reward = 10*x_vel + abs(y_vel)

        Map_array = ["curved_map1", "curved_map2", "curved_map3", "curved_map4"]

        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]
            for j in range(len(Map_array)):
                if (str(self.sim.model.geom_id2name(sim_contact.geom1)) == Map_array[j]):
                    print("Collision! : Reward -= 10")
                    reward -= 10
                    break
                elif (str(self.sim.model.geom_id2name(sim_contact.geom2)) == Map_array[j]):
                    print("Collision! : Reward -= 10")
                    reward -= 10
                    break


        for i in range(self.sim.data.ncon):
            sim_contact = self.sim.data.contact[i]

            if(str(self.sim.model.geom_id2name(sim_contact.geom1)) == "Goal"):
                print("Goal IN! : Reward = 5000")
                reward = 5000
                break

            elif(str(self.sim.model.geom_id2name(sim_contact.geom2)) == "Goal"):
                print("Goal IN! : Reward = 5000")
                reward = 5000
                break


        done = self.done
        observation = self._get_obs()
        info = {
               
            'total reward' : reward
        }

        return observation, reward, done, info

    def _get_obs(self):

        ## 1. For Training
        #camera_data = np.array(self.render("rgb_array", 148, 148, 2))
        #CHW = np.transpose(camera_data, (2, 0, 1))

        ## If you wanna check the input image
        #plt.imshow(camera_data)
        #plt.show()

        ## 2. For rendering check
        data = self._get_viewer("rgb_array").read_pixels(148, 148, depth=False)
        CHW = np.transpose(data[::-1, :, :] , (2, 0, 1))

        obs_dct = {}
        obs_dct['image'] = np.array(CHW)/255.0
        obs_dct['vector'] = self.state_vector()[6:24]

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


