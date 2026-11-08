# unitree_sdk2_python
Python interface for unitree_sdk2.

# Installation
## Dependencies
- python>=3.8
- cyclonedds==0.10.2
- numpy
- opencv-python

## Install unitree_sdk2_python
Run in a terminal:
```bash
cd ~
sudo apt install python3-pip
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip3 install -e .
```
## FAQ
##### 1. Error when running `pip3 install -e .`
```bash
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```
This error indicates that the cyclonedds path was not found. Build and install cyclonedds first:
```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x 
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
```
Enter the unitree_sdk2_python directory, set `CYCLONEDDS_HOME` to the cyclonedds build path, and install unitree_sdk2_python:
```bash
cd ~/unitree_sdk2_python
export CYCLONEDDS_HOME="~/cyclonedds/install"
pip3 install -e .
```

For details, see:
https://pypi.org/project/cyclonedds/#installing-with-pre-built-binaries

# Usage
The Python SDK2 interface follows unitree_sdk2 and retrieves robot state or sends controls through request/response calls and publish/subscribe topics. Examples are in `/example`. Before running them, configure the robot's network connection using https://support.unitree.com/home/zh/developer/Quick_start.
## DDS Communication
Run in a terminal:
```bash
python3 ./example/helloworld/publisher.py
```
Open a new terminal and run:
```bash
python3 ./example/helloworld/subscriber.py
```
The terminal displays the received data. The data exchanged by `publisher.py` and `subscriber.py` is defined in `user_data.py`; customize these structures as needed.

## High-Level State and Control
High-level data structures and controls follow unitree_sdk2. For details, see https://support.unitree.com/home/zh/developer/sports_services.
### High-Level State
Run in a terminal:
```bash
python3 ./example/high_level/read_highstate.py enp2s0
```
Here, `enp2s0` is the network interface connected to the robot. Replace it with your actual interface name.
### High-Level Control
Run in a terminal:
```bash
python3 ./example/high_level/sportmode_test.py enp2s0
```
Here, `enp2s0` is the network interface connected to the robot. Replace it with your actual interface name.
The example provides several test methods; enable the method you need:
```python
test.StandUpDown() # Stand up and lie down.
# test.VelocityMove() # Velocity control.
# test.BalanceAttitude() # Attitude control.
# test.TrajectoryFollow() # Trajectory tracking.
# test.SpecialMotions() # Special motions.

```
## Low-Level State and Control
Low-level data structures and controls follow unitree_sdk2. For details, see https://support.unitree.com/home/zh/developer/Basic_services.
### Low-Level State
Run in a terminal:
```bash
python3 ./example/low_level/lowlevel_control.py enp2s0
```
Replace `enp2s0` with the network interface connected to the robot. The program prints the front-right hip joint state, IMU data, and battery voltage.

### Low-Level Motor Control
First disable the high-level motion service (sport_mode) in the app to prevent conflicting commands.
Run in a terminal:
```bash
python3 ./example/low_level/lowlevel_control.py enp2s0
```
Replace `enp2s0` with the network interface connected to the robot. The rear-left hip joint holds an angle of 0 (kp=10 and kd=1 for safety), while the rear-left calf joint continuously applies 1 Nm of torque.

## Remote Controller State
Run in a terminal:
```bash
python3 ./example/wireless_controller/wireless_controller.py enp2s0
```
Here, `enp2s0` is the network interface connected to the robot. Replace it with your actual interface name.
The terminal prints each button's state. For button definitions and data structures, see https://support.unitree.com/home/zh/developer/Get_remote_control_status.

## Front Camera
Use OpenCV to access the front camera (run on a system with a graphical display; press ESC to quit):
```bash
python3 ./example/front_camera/camera_opencv.py enp2s0
```
Here, `enp2s0` is the network interface connected to the robot. Replace it with your actual interface name.

## Obstacle Avoidance Toggle
```bash
python3 ./example/obstacles_avoid_switch/obstacles_avoid_switch.py enp2s0
```
Replace `enp2s0` with the network interface connected to the robot. The robot repeatedly enables and disables obstacle avoidance. For details, see https://support.unitree.com/home/zh/developer/ObstaclesAvoidClient.

## Light and Volume Control
```bash
python3 ./example/vui_client/vui_client_example.py enp2s0
```
Replace `enp2s0` with the network interface connected to the robot. The robot repeatedly adjusts volume and light brightness. For details, see https://support.unitree.com/home/zh/developer/VuiClient.
