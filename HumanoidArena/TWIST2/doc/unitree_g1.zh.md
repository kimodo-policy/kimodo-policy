# Deployment on Physical Robots

This code deploys trained policies on physical robots. Supported robots include Unitree G1, H1, and H1_2.

| G1 | H1 | H1_2 |
|--- | --- | --- |
| [![real_g1](https://oss-global-cdn.unitree.com/static/78c61459d3ab41448cfdb31f6a537e8b.GIF)](https://oss-global-cdn.unitree.com/static/0818dcf7a6874b92997354d628adcacd.mp4) | [![real_h1](https://oss-global-cdn.unitree.com/static/fa07b2fd2ad64bb08e6b624d39336245.GIF)](https://oss-global-cdn.unitree.com/static/ea0084038d384e3eaa73b961f33e6210.mp4) | [![real_h1_2](https://oss-global-cdn.unitree.com/static/a88915e3523546128a79520aa3e20979.GIF)](https://oss-global-cdn.unitree.com/static/12d041a7906e489fae79d55b091a63dd.mp4) |

## Launch Usage

```bash
python deploy_real.py {net_interface} {config_name}
```

- `net_interface`: Name of the network interface connected to the robot, such as `enp3s0`.
- `config_name`: Configuration filename under `deploy/deploy_real/configs/`, such as `g1.yaml`, `h1.yaml`, or `h1_2.yaml`.

## Startup Procedure

### 1. Power On the Robot

Power on the robot while it is supported by a suspension rig, and wait for it to enter zero-torque mode.

### 2. Enter Debug Mode

With the robot in zero-torque mode, press `L2+R2` on the controller to enter debug mode. The joints are damped in this mode.

### 3. Connect to the Robot

Connect the computer to the robot with an Ethernet cable and configure the network as follows.

<img src="https://doc-cdn.unitree.com/static/2023/9/6/0f51cb9b12f94f0cb75070d05118c00a_980x816.jpg" width="400px">

Use `ifconfig` to identify the network interface connected to the robot. Record its name for the launch command.

<img src="https://oss-global-cdn.unitree.com/static/b84485f386994ef08b0ccfa928ab3830_825x484.png" width="400px">

### 4. Start the Program

For a G1 connected through the `enp3s0` interface, start the program with the following command.

```bash
python deploy/deploy_real/deploy_real.py enp3s0 g1.yaml
```

#### 4.1 Zero-Torque State

After startup, the joints remain in the zero-torque state. Carefully move a joint by hand to confirm this state.

#### 4.2 Default Pose

In the zero-torque state, press `start` on the controller to move the robot to its default joint configuration.

Once the default pose is reached, slowly lower the suspension rig until the feet touch the ground.

#### 4.3 Motion Control Mode

When ready, press `A` on the controller. The robot will step in place. Once stable, gradually loosen the suspension to allow some movement.

Use the controller joysticks to command robot motion.
Move the left joystick forward/backward to control x velocity.
Move the left joystick left/right to control y velocity.
Move the right joystick left/right to control yaw velocity.

#### 4.4 Exit Control

In motion control mode, pressing `select` enters damping mode and exits the program; the robot may fall. Alternatively, stop the program with `Ctrl+C` in the terminal.

> Note:
> 
> This deployment is an example, not a stable production controller. Avoid applying disturbances during control, and exit control promptly if unexpected behavior occurs to reduce the risk of injury or damage.

## Video Tutorials

G1 deployment: [https://oss-global-cdn.unitree.com/static/ff70a257ddf34adbb67733d8a90b24d4.mp4](https://oss-global-cdn.unitree.com/static/ff70a257ddf34adbb67733d8a90b24d4.mp4)

H1 deployment: [https://oss-global-cdn.unitree.com/static/6dae1756f4214409a5ced7386ca011ae.mp4](https://oss-global-cdn.unitree.com/static/6dae1756f4214409a5ced7386ca011ae.mp4)

H1_2 deployment: [https://oss-global-cdn.unitree.com/static/9d61a1470d3e4b9a9af6d131324fcb94.mp4](https://oss-global-cdn.unitree.com/static/9d61a1470d3e4b9a9af6d131324fcb94.mp4)