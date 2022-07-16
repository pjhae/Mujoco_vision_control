# Mujoco_vision_control

## Implementation of vision-based robot control using CNN Feature Extraction and RL.

### 0. Requirements
 Installation : GYM, MUJOCO, stablebaselines3 + (Linux)

  1. Move to

    YOUR_PATH/python3.X/site-packages/gym/

  2. Clone the repository in

    YOUR_PATH/python3.X/site-packages/gym/envs/

### 1. Problem Statement

Our main goal is to verify that robot control can be implemented in a simulated environment using vision-based RL algorithms.
For this purpose, two tasks are set up.
![image](https://user-images.githubusercontent.com/74540268/179348883-e2e23c23-31f5-40ec-bd59-769db91b549f.png)


### 2. Hardware 3D design
URDF, xml : All links and joints are manually reverse engineered using assembly file from [Arcbotics](http://arcbotics.com/products/hexy/) 
#### 2.1. [Solidworks] re-assembly : 
![hexy_SW6](https://user-images.githubusercontent.com/74540268/169776703-d9660b52-a81e-4ba5-ab9a-c01d76072a12.PNG)


#### 2.2. [URDF] Hitbox design : 
![hexy_heat2](https://user-images.githubusercontent.com/74540268/169944721-46a89900-eaed-4b17-b6cb-a4496fd48ab6.PNG)


### 3. MUJOCO camera sensor
![image](https://user-images.githubusercontent.com/74540268/179348919-d6c75e1e-551c-4213-89dd-9feae3f25a6c.png)


