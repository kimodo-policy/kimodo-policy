# Teleop Pipeline


1.1 Start the G1 robot
1.2 `bash docker_neck.sh` for starting neck server
1.3 replug the ZED MINI camera to ensure connection
1.4 `bash docker_zed.sh` for starting orin zed sender
1.5 Start to listen to ZED MINI camera in VR (you should now see the camera feed in the VR)


2.1 Wear motion trackers; wear controllers on the wrists
2.2 Start the VR
2.3 Calibrate the whole-body motions
2.4 Enter XRobot APP
2.5 Connect to IP of my ubuntu
2.6 Start to streaming whole-body data and hand data
2.7 start teleop in mujoco
```bash
bash teleop_motion_gen.sh
```
2.8 test sim2sim first
```bash
bash sim2sim.sh
```



3.1 Start data recording
```bash
bash teleop_data_record.sh
```

3.2 start sim2real
```bash
bash sim2real.sh
```

# Controller buttons

teleop:

    Right A: Start/pause teleoperation
    Left X: Exit teleoperation and enter the default pose

    Right index grip: Close the right hand
    Right grip: Open the right hand

    Left index grip: Close the left hand
    Left grip: Open the left hand

    Right B: Reduce the size of the streamed RGB view in VR

    Left axis click: Robot emergency stop
    

motion gen:

    Right axis click: Enter/exit motion generation mode

    Left joystick: Control robot x motion in motion generation mode
    Right joystick: Control robot yaw in motion generation mode

data record:

    Left Y: Start/pause recording

    Left axis click: Exit recording




# APP Setting

Pull from PICO:
```bash
adb pull /sdcard/Android/data/com.xrobotoolkit.client/files/video_source.yml video_source.yml
```

Push to PICO:
```bash
adb push video_source.yml /sdcard/Android/data/com.xrobotoolkit.client/files/video_source.yml
```

