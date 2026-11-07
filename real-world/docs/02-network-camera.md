# 2. Networking, PICO, and the Robot Camera

Run workstation commands after sourcing `env.sh`. G1 and PICO use separate wired interfaces and subnets. Wi-Fi or another uplink connects the workstation to the model server.

## 2.1 Identify and configure interfaces

```bash
ip -br link
ip -br addr
ip route
```

Set the actual interface names in `local.env`. Choose unused workstation addresses in the appropriate device subnets:

```bash
export G1_WORKSTATION_CIDR='your_workstation_address/prefix'
export PICO_WORKSTATION_CIDR='your_workstation_pico_address/prefix'
sudo nmcli connection add type ethernet con-name kimodo-g1 \
  ifname "$ROBOT_INTERFACE" ipv4.method manual ipv4.addresses "$G1_WORKSTATION_CIDR" \
  ipv4.never-default yes ipv6.method disabled connection.autoconnect no
sudo nmcli connection add type ethernet con-name kimodo-pico \
  ifname "$PICO_INTERFACE" ipv4.method manual ipv4.addresses "$PICO_WORKSTATION_CIDR" \
  ipv4.never-default yes ipv6.method disabled connection.autoconnect no
sudo nmcli connection up kimodo-g1
sudo nmcli connection up kimodo-pico
ip route get "$CAMERA_HOST"
ip -4 -o addr show dev "$PICO_INTERFACE"
```

Use `nmcli connection modify` instead of `add` if a profile already exists. The G1 route must use `ROBOT_INTERFACE`; neither device interface should replace your uplink's default route.

Manual configuration does not assign an address to PICO. If PICO obtains its address through DHCP and there is no other DHCP server, enable sharing on the **PICO interface only**:

```bash
sudo nmcli connection modify kimodo-pico ipv4.method shared
sudo nmcli connection down kimodo-pico
sudo nmcli connection up kimodo-pico
ip -4 -o addr show dev "$PICO_INTERFACE"
```

In the PICO application's Connect field, enter the **workstation's PICO-interface IPv4 address**, without `/prefix`. Do not enter the model-server address or PICO's own address. Permit required traffic for the device subnet in your firewall. The PC Service SDK uses port 60061; consult the matched XRT version for any additional ports.

## 2.2 Robot camera service

Keep an existing compatible composed-camera service if it already works; proceed to section 2.3. To install the provided RealSense entry point, the robot needs its own architecture-compatible Python 3.10 environment and working librealsense/pyrealsense2 installation.

From the workstation:

```bash
export G1_SSH=your_g1_ssh_alias
ssh "$G1_SSH" 'mkdir -p ~/kimodo-camera/gear_sonic'
rsync -a "$WBC_ROOT/gear_sonic/camera" "$G1_SSH:kimodo-camera/gear_sonic/"
rsync -a "$WBC_ROOT/gear_sonic/__init__.py" "$G1_SSH:kimodo-camera/gear_sonic/"
rsync -a "$WBC_ROOT/gear_sonic/scripts/run_realsense_camera_server.py" "$G1_SSH:kimodo-camera/server.py"
```

On the robot, activate the camera environment:

```bash
cd "$HOME/kimodo-camera"
python -c 'import pyrealsense2, cv2, numpy, msgpack, msgpack_numpy, zmq; print("camera imports OK")'
python -c 'import pyrealsense2 as rs; print([d.get_info(rs.camera_info.name) for d in rs.context().query_devices()])'
PYTHONPATH="$PWD" python server.py --port 5555 --fps 30
```

Install camera dependencies for the robot's CPU architecture. Never copy the workstation's x86 native libraries to an ARM64 robot. Robot-side dependency versions have not yet been verified. Validate a foreground service before installing a systemd unit; avoid multiple services competing for the camera. The RealSense command is not an OAK camera driver: use the appropriate existing composed-camera service for OAK hardware.

## 2.3 Validate camera and clocks

```bash
bash collection/bin/check-network
bash collection/bin/check-camera
```

Require fresh image frames, not merely an open TCP port. The observed input was 640×480 at approximately 30 Hz. Match camera placement, RGB channel order, and exposure to the training data. Check `timedatectl status` on workstation and robot; clock offsets can invalidate freshness/skew checks.

## 2.4 Connect PICO

In a dedicated workstation terminal:

```bash
bash collection/bin/xrt-service
```

Calibrate the headset and operator height, open the matching XRoboToolkit application, and connect to the workstation address from section 2.1. Once connected, start the manager in the next chapter. PC Service 1.0.0.0 and Python SDK 1.0.2 are recorded; the matching headset APK version remains to be confirmed.
