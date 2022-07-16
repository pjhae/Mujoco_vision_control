# Mujoco_vision_control

## Implementation of vision-based robot control using CNN Feature Extraction and RL.

### 0. Requirements
 Installation : GYM, MUJOCO, stablebaselines3 + (Linux)

  1. Move to

    YOUR_PATH/python3.X/site-packages/gym/

  2. Clone this repository in

    YOUR_PATH/python3.X/site-packages/gym/envs/

##
### 1. Problem Statement

Our main goal is to verify that robot control can be implemented in a simulated environment using vision-based RL algorithms.

For this purpose, two tasks are set up.
![image](https://user-images.githubusercontent.com/74540268/179348883-e2e23c23-31f5-40ec-bd59-769db91b549f.png)

##
### 2. Hardware 3D design
URDF, xml : All links and joints are manually reverse engineered using assembly file from [Arcbotics](http://arcbotics.com/products/hexy/) 

![hexy_heat2](https://user-images.githubusercontent.com/74540268/169944721-46a89900-eaed-4b17-b6cb-a4496fd48ab6.PNG)

##
### 3. MUJOCO camera sensor
![image](https://user-images.githubusercontent.com/74540268/179350874-c0dfab51-01d5-4d2c-8a16-8684bcc4f9b1.png)

For mounting Camera on Robot Model, you can see the file in gym/mujoco/assets/Hexy_ver_2.3/assets

To get RGB data from camera for observation, you can see the file in gym/mujoco/hexy_v5,6.py


##
### 4. Policy Net

![image](https://user-images.githubusercontent.com/74540268/179349101-6eb8b4ff-d24e-486e-99dd-2e28ca9d6620.png)


• Input : Image RGB data + current motor angle


• Output : Desired motor angle

• RL algorithm : PPO


##
### 5. Training env
you can see the code for specific MDP setting(S,A,R..) info

    YOUR_PATH/python3.X/site-packages/gym/envs/mujoco/


**hexy_v4.py** : Just walking without vision infomation (using only current qpos as observation)

**hexy_v5.py** : *Task 1*

**hexy_v6.py** : *Task 2*

**hexy_v7.py** : One-hot vector verification for *Task 2*


##
### 6. Results

