# Episode Reader Guide

Utilities for reading and visualizing recorded TWIST2 data.

## Features

- ✅ Load episode data (front and world cameras supported)
- ✅ Generate front-camera or world-camera videos
- ✅ Overlay joint keypoints on world-camera images
- ✅ Access recorded SMPLX data, states, actions, and other fields
- ✅ Process a frame range using start_frame and end_frame

## Quick Start

### 1. Command Line

#### Inspect Episode Information
```bash
python3 tools/episode_reader.py /path/to/episode_0001 --info
```

#### Generate a Front-Camera Video
```bash
python3 tools/episode_reader.py /path/to/episode_0001 \
    --create-video output/front.mp4 \
    --camera front
```

#### Generate a World-Camera Video
```bash
python3 tools/episode_reader.py /path/to/episode_0001 \
    --create-video output/world.mp4 \
    --camera world
```

#### Generate a World-Camera Video with Joint Keypoints
```bash
python3 tools/episode_reader.py /path/to/episode_0001 \
    --visualize-keypoints output/world_with_keypoints.mp4
```

#### Select a Frame Range
```bash
python3 tools/episode_reader.py /path/to/episode_0001 \
    --visualize-keypoints output/partial.mp4 \
    --start-frame 0 \
    --end-frame 300
```

### 2. Python API

```python
from episode_reader import EpisodeReader

# Load data
reader = EpisodeReader('/path/to/episode_0001')

# Print detailed information
reader.print_info()

# Generate videos
reader.create_video('front.mp4', camera='front')
reader.create_video('world.mp4', camera='world')

# Visualize joint keypoints
reader.visualize_keypoints_on_world_cam('world_with_keypoints.mp4')

# Access data
frame_data = reader.get_frame(100)
front_img = reader.get_image(100, 'front')  # RGB format
world_img = reader.get_image(100, 'world')
keypoints = reader.get_keypoints(100)
smplx = reader.get_smplx_data(100)
state = reader.get_state_body(100)
action = reader.get_action_body(100)
```

### 3. Run the Examples

```bash
# Run all examples
python3 tools/example_usage.py --example all

# Run a specific example
python3 tools/example_usage.py --example basic      # Basic usage
python3 tools/example_usage.py --example videos    # Generate videos
python3 tools/example_usage.py --example keypoints # Visualize keypoints
python3 tools/example_usage.py --example data      # Access data
python3 tools/example_usage.py --example partial   # Generate a partial video
```

## Data Structure

Episode directory layout:
```
episode_0001/
├── data.json          # Metadata and all non-image data
├── front_rgb/         # Front-camera image sequence
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── world_rgb/         # World-camera image sequence (optional)
    ├── 000000.jpg
    ├── 000001.jpg
    └── ...
```

Fields in data.json:
- `info`: Recording metadata (version, date, fps, etc.)
- `text`: Task descriptions (goal, desc, steps)
- `data`: List of frames; each frame contains:
  - `front_rgb`: Front-camera image path
  - `world_rgb`: World-camera image path (may be empty)
  - `world_camera_joint_keypoints`: 2D joint coordinates
  - `smplx_data`: SMPLX human body pose data
  - `state_body`, `state_hand_left`, `state_hand_right`, `state_neck`: State data
  - `action_body`, `action_hand_left`, `action_hand_right`, `action_neck`: Action data
  - Timestamps and other metadata

## API Reference

### EpisodeReader Class

#### Initialization
```python
reader = EpisodeReader(episode_path: str)
```

#### Main Methods

**Read data:**
- `get_frame(idx: int) -> Dict`: Get all data for a frame
- `get_image(idx: int, camera: str) -> np.ndarray`: Get an image in RGB format
- `get_keypoints(idx: int) -> List`: Get joint coordinates
- `get_smplx_data(idx: int) -> Dict`: Get SMPLX data
- `get_state_body(idx: int) -> List`: Get body state
- `get_action_body(idx: int) -> List`: Get body actions
- `get_state_hand(idx: int, hand: str) -> List`: Get hand state
- `get_action_hand(idx: int, hand: str) -> List`: Get hand actions

**Generate videos:**
- `create_video(output_path, camera, fps, start_frame, end_frame) -> bool`: Generate a video
- `visualize_keypoints_on_world_cam(output_path, fps, ...) -> bool`: Generate a video with joint keypoints

**Other methods:**
- `print_info()`: Print detailed information
- `__len__()`: Return the frame count

#### Properties
- `frames`: All frame data
- `info`: Recording metadata
- `text`: Task descriptions
- `has_front_cam`: Whether front-camera data is available
- `has_world_cam`: Whether world-camera data is available
- `fps`: Frame rate

## Tests

Test data:
```bash
/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001
```

Quick test:
```bash
# Inspect episode information
python3 tools/episode_reader.py data/demo_20260114_222032/episode_0001 --info

# Generate a test video (first 30 frames)
python3 tools/episode_reader.py data/demo_20260114_222032/episode_0001 \
    --visualize-keypoints test_output.mp4 \
    --start-frame 0 \
    --end-frame 30
```

## Notes

1. **The world camera is optional:** the code detects missing data and reports it.
2. **Image format:** `get_image()` returns RGB; convert to BGR when required by OpenCV.
3. **Joint coordinates:** a keypoint may be None when outside the camera view.
4. **Video encoding:** mp4v is the default; use the codec argument to change it.

## Troubleshooting

**Q: The program reports "World camera not available".**
A: This episode has no world-camera data; only front-camera functions are available.

**Q: The generated video will not play.**
A: Try another codec, such as 'XVID' or 'H264'.

**Q: Joint keypoints are not displayed.**
A: Check whether world_camera_joint_keypoints is None. Some frames may lack keypoints.

## Related Files

- `episode_reader.py`: Main data reader class
- `example_usage.py`: Usage examples
- `../deploy_real/server_data_record_from_shm.py`: Recording script
- `../deploy_real/data_utils/episode_writer.py`: Data writer class
