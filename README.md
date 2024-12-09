# robotarm_pick_and_place
This project is my small project to learn how to use realsense camera d435 and 3D printed robot arm to operate a pick and place from scratch.

## Starting the Robot

First, cd into the project's directory.
```
cd ~/myrobot
```

If a package has been modified or if this is your first time testing the project, run:
```
catkin_make
```

Everytime you want to start the robot, turn it on and run the following commands:
```
source devel/setup.bash
roslaunch vision_module execute_all.launch
```
*Note: if you want to run just the simulation without having the arm connected, change the real_arm parameter to false.*