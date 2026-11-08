#!/bin/bash

# TWIST2+IsaacLab enhanced data recording system - launcher
#
# Automatically start all required components
# Usage:bash start_enhanced_recording.sh

set -e  # Exit immediately on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print a colored message
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
}

# Check dependencies
check_dependencies() {
    print_header "Checking system dependencies"

    # Check Redis
    if ! command -v redis-cli &> /dev/null; then
        print_error "Redis is not installed; install Redis first"
        exit 1
    fi

    if ! redis-cli ping &> /dev/null; then
        print_error "Redis is not running; starting it..."
        sudo systemctl start redis || redis-server --daemonize yes
        sleep 2
        if redis-cli ping &> /dev/null; then
            print_success "Redis started successfully"
        else
            print_error "Failed to start Redis"
            exit 1
        fi
    else
        print_success "Redis is running"
    fi

    # Check the Python environment
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed"
        exit 1
    fi
    print_success "Python3 is installed: $(python3 --version)"
}

# Configure parameters
configure_params() {
    print_header "Configuration"

    # Default parameters
    DEFAULT_ROBOT_IP="192.168.123.164"
    DEFAULT_TASK_NAME="demo_$(date +%Y%m%d_%H%M%S)"
    DEFAULT_FREQUENCY="30"
    DEFAULT_DATA_FOLDER="./twist2_demonstration_smplx"

    # Interactive configuration or use defaults
    read -p "Robot IP address [${DEFAULT_ROBOT_IP}]: " ROBOT_IP
    ROBOT_IP=${ROBOT_IP:-$DEFAULT_ROBOT_IP}

    read -p "Task name [${DEFAULT_TASK_NAME}]: " TASK_NAME
    TASK_NAME=${TASK_NAME:-$DEFAULT_TASK_NAME}

    read -p "Recording frequency (Hz) [${DEFAULT_FREQUENCY}]: " FREQUENCY
    FREQUENCY=${FREQUENCY:-$DEFAULT_FREQUENCY}

    read -p "Data output directory [${DEFAULT_DATA_FOLDER}]: " DATA_FOLDER
    DATA_FOLDER=${DATA_FOLDER:-$DEFAULT_DATA_FOLDER}

    echo ""
    print_info "Configuration summary:"
    echo "  Robot IP: ${ROBOT_IP}"
    echo "  Task name: ${TASK_NAME}"
    echo "  Recording frequency: ${FREQUENCY} Hz"
    echo "  Data directory: ${DATA_FOLDER}"
    echo ""

    read -p "Continue? (y/n) [y]: " CONFIRM
    CONFIRM=${CONFIRM:-y}
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        print_info "Cancelled"
        exit 0
    fi
}

# Check required scripts
check_scripts() {
    print_header "Checking script files"

    TWIST2_DIR="/home/hcl4070-1/Desktop/taowen/projects/TWIST2"
    ISAACLAB_DIR="/home/hcl4070-1/Desktop/taowen/projects/isaaclab_twist2_g1"

    # Check the TWIST2 teleoperation script
    TELEOP_SCRIPT="${TWIST2_DIR}/deploy_real/xrobot_teleop_to_robot_w_hand.py"
    if [[ ! -f "$TELEOP_SCRIPT" ]]; then
        print_error "Teleoperation script not found: $TELEOP_SCRIPT"
        exit 1
    fi
    print_success "Teleoperation script: $TELEOP_SCRIPT"

    # Check the data recording script
    RECORD_SCRIPT="${TWIST2_DIR}/deploy_real/server_data_record_with_third_smplx_qpos.py"
    if [[ ! -f "$RECORD_SCRIPT" ]]; then
        print_error "Recording script not found: $RECORD_SCRIPT"
        exit 1
    fi
    print_success "Recording script: $RECORD_SCRIPT"

    # Check the IsaacLab sim_main entry point
    ISAACLAB_SCRIPT="${ISAACLAB_DIR}/sim_main.py"
    if [[ ! -f "$ISAACLAB_SCRIPT" ]]; then
        print_warning "IsaacLab script not found: $ISAACLAB_SCRIPT"
        print_warning "Start the IsaacLab simulation manually"
    else
        print_success "IsaacLab script: $ISAACLAB_SCRIPT"
    fi
}

# Check whether IsaacLab is running
check_isaaclab() {
    print_header "Checking IsaacLab status"

    # Check the image-server port
    if nc -zv localhost 5555 2>&1 | grep -q succeeded; then
        print_success "Front camera server (port 5555) is running"
    else
        print_warning "Front camera server (port 5555) was not detected"
        print_info "Ensure the IsaacLab simulation and image server are running"
    fi

    if nc -zv localhost 5556 2>&1 | grep -q succeeded; then
        print_success "World camera server (port 5556) is running"
    else
        print_warning "World camera server (port 5556) was not detected"
        print_info "Ensure the IsaacLab simulation and image server are running"
    fi

    echo ""
    print_info "If IsaacLab is not running, start it in another terminal:"
    echo "  cd ${ISAACLAB_DIR}"
    echo "  ./run.sh"
    echo ""
    read -p "Is IsaacLab already running? (y/n) [y]: " ISAACLAB_RUNNING
    ISAACLAB_RUNNING=${ISAACLAB_RUNNING:-y}
    if [[ ! "$ISAACLAB_RUNNING" =~ ^[Yy]$ ]]; then
        print_error "Start the IsaacLab simulation first"
        exit 1
    fi
}

# Start the teleoperation system
start_teleop() {
    print_header "Starting TWIST2 teleoperation"

    cd "${TWIST2_DIR}"

    # Check whether it is already running
    if pgrep -f "xrobot_teleop_to_robot_w_hand.py" > /dev/null; then
        print_success "Teleoperation is already running"
        return 0
    fi

    print_info "Starting teleoperation in a new terminal..."

    # Launch with gnome-terminal or xterm
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal -- bash -c "
            cd ${TWIST2_DIR};
            echo 'Starting TWIST2 teleoperation...';
            python deploy_real/xrobot_teleop_to_robot_w_hand.py --robot unitree_g1;
            read -p 'Press Enter to close...'
        " &
        TELEOP_PID=$!
        print_success "Teleoperation is starting in a new terminal..."
    elif command -v xterm &> /dev/null; then
        xterm -e "
            cd ${TWIST2_DIR};
            echo 'Starting TWIST2 teleoperation...';
            python deploy_real/xrobot_teleop_to_robot_w_hand.py --robot unitree_g1;
            read -p 'Press Enter to close...'
        " &
        TELEOP_PID=$!
        print_success "Teleoperation is starting in a new terminal..."
    else
        print_warning "Could not open a new terminal automatically"
        print_info "Run the following in a new terminal:"
        echo "  cd ${TWIST2_DIR}"
        echo "  python deploy_real/xrobot_teleop_to_robot_w_hand.py --robot unitree_g1"
        echo ""
        read -p "Press Enter once startup is complete..."
    fi

    # Wait for the teleoperation system to start
    print_info "Waiting for teleoperation to start..."
    sleep 5

    # Check whether SMPL-X data is available
    for i in {1..10}; do
        if redis-cli exists smplx_data_unitree_g1_with_hands | grep -q 1; then
            print_success "SMPLX data is ready"
            break
        fi
        if [ $i -eq 10 ]; then
            print_warning "SMPLX data was not detected; continuing..."
        fi
        sleep 1
    done
}

# Start data recording
start_recording() {
    print_header "Starting the data recorder"

    cd "${TWIST2_DIR}"

    print_info "Startup parameters:"
    echo "  --robot_ip ${ROBOT_IP}"
    echo "  --task_name ${TASK_NAME}"
    echo "  --frequency ${FREQUENCY}"
    echo "  --data_folder ${DATA_FOLDER}"
    echo ""

    print_info "Controls:"
    echo "  Left controller key_two    : Start/stop recording"
    echo "  Left controller axis_click : Exit"
    echo ""

    print_success "Starting recording..."

    python deploy_real/server_data_record_with_third_smplx_qpos.py \
        --robot_ip "${ROBOT_IP}" \
        --task_name "${TASK_NAME}" \
        --frequency "${FREQUENCY}" \
        --data_folder "${DATA_FOLDER}"
}

# Cleanup function
cleanup() {
    print_header "Cleaning up resources"

    # Add cleanup logic here
    # For example, stop the processes started by this script

    print_success "Cleanup complete"
}

# Run main function
main() {
    # Handle Ctrl+C
    trap cleanup EXIT

    print_header "TWIST2+IsaacLab enhanced data recorder"

    # 1. Check dependencies
    check_dependencies

    # 2. Configure parameters
    configure_params

    # 3. Check scripts
    check_scripts

    # 4. Check IsaacLab
    check_isaaclab

    # 5. Start teleoperation (optional)
    read -p "Start TWIST2 teleoperation? (y/n) [y]: " START_TELEOP
    START_TELEOP=${START_TELEOP:-y}
    if [[ "$START_TELEOP" =~ ^[Yy]$ ]]; then
        start_teleop
    else
        print_info "Skipping teleoperation startup (assumed to be running)"
    fi

    # 6. Start data recording
    start_recording

    print_success "All components have stopped"
}

# Run main function
main "$@"
