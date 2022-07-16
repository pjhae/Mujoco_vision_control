# Mujoco_vision_control

## Implementation of vision-based robot control using CNN Feature Extraction and RL.

URDF, xml : All links and joints are manually reverse engineered using assembly file from [Arcbotics](http://arcbotics.com/products/hexy/) 

### 0. Requirements
 Installation : GYM, MUJOCO, stablebaselines3 + (Linux)

  1. Move to

    YOUR_PATH/raisimLib/raisimGymTorch/raisimGymTorch/

  2. Clone the repository in

    YOUR_PATH/raisimLib/raisimGymTorch/raisimGymTorch/

  3. Move [hexapod] directory to

    YOUR_PATH/raisimLib/rsc/

  4. Set up environment

    cd YOUR_PATH/raisimLib/raisimGymTorch/
    python setup.py develop
  
  5. Train/Test

    cd YOUR_PATH/raisimLib/raisimGymTorch/env/envs/hexapod_command_locomotion
    (Training) python runner.py
    (Test) python tester.py -w [model_PATH]

  6. (optional) Deployment

    cd YOUR_PATH/raisimLib/raisimGymTorch/env/envs/hexapod_command_locomotion
    python deploy.py -w [model_PATH]


## 
### 1. MiniCheetah

#### 1.1. [Solidworks] re-assembly process :
**Video : [Link](https://youtu.be/irAciVODtpI)**


#### 1.2. [URDF] Hitbox design :
 using basic shape (Box, cylinder, sphere..)
![Hitbox](https://user-images.githubusercontent.com/74540268/169758719-4ecca46f-24fb-4cca-b3a1-0682afbeb4c0.PNG)


#### 1.3. [MUJOCO,GYM,sb3] Training results :
you can see the code for specific MDP setting(S,A,R..) info
   ![ezgif com-gif-maker (5)](https://user-images.githubusercontent.com/74540268/169943131-9eccfde6-9127-4d72-a83f-b7c9df6ee2b6.gif)

  
## 
## 
 

### 2. Hexapod

#### 2.1. [Solidworks] re-assembly : 
![hexy_SW6](https://user-images.githubusercontent.com/74540268/169776703-d9660b52-a81e-4ba5-ab9a-c01d76072a12.PNG)



#### 2.2. [URDF] Hitbox design : 
![hexy_heat2](https://user-images.githubusercontent.com/74540268/169944721-46a89900-eaed-4b17-b6cb-a4496fd48ab6.PNG)




#### 2.3. [MUJOCO,GYM,sb3] Training results : 
you can see the code for specific MDP setting(S,A,R..) info
   ![hexy2](https://user-images.githubusercontent.com/74540268/169965058-dae06b5d-aa1c-476b-943a-8457845638db.gif)


## 

