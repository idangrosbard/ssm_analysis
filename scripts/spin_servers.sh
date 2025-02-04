#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ZENML_PORT=8237
MLFLOW_PORT=8239
MACHINE_NAME="$(hostname).cs.tau.ac.il"
MACHINE_IP="$(hostname -I | awk '{print $1}')"

# Define the base directory (one level up from the script directory)
PROJECT_BASE="$(dirname "$SCRIPT_DIR")"

# Assert that the base directory name is "ssm_analysis"
EXPECTED_BASE_DIR_NAME="ssm_analysis"
ACTUAL_BASE_DIR_NAME="$(basename "$PROJECT_BASE")"

ZENML_PROCESS_PATTERN="zenml.services"
MLFLOW_PROCESS_PATTERN="mlflow.server"

if [ "$ACTUAL_BASE_DIR_NAME" != "$EXPECTED_BASE_DIR_NAME" ]; then
    echo "Error: The base directory name must be '$EXPECTED_BASE_DIR_NAME'. Found '$ACTUAL_BASE_DIR_NAME' instead."
    exit 1
fi


# Define custom directories for ZenML and MLflow
ZENML_DIR="$PROJECT_BASE/.zen"
export ZENML_CONFIG_PATH="$ZENML_DIR/"
MLFLOW_DIR="$PROJECT_BASE/.mlflow"

# Create directories if they don't exist
mkdir -p "$ZENML_DIR"
mkdir -p "$MLFLOW_DIR"

# Function to start servers
start_servers() {
    echo "Starting ZenML and MLflow servers..."
    echo "Running on $MACHINE_NAME ($MACHINE_IP)"
    echo "$MACHINE_NAME" > "$ZENML_DIR/machine_name.txt"
    echo "$MACHINE_IP" > "$ZENML_DIR/machine_ip.txt"
    
    # Check if ZenML server is already running
    if ! pgrep -f "$ZENML_PROCESS_PATTERN" > /dev/null; then
        echo "Starting ZenML server in $ZENML_DIR..."
        cmd="zenml login --local --ip-address $MACHINE_IP --port $ZENML_PORT &> $ZENML_DIR/zenml.log &"
        echo "$cmd"
        eval "$cmd"
        echo "ZenML server started."
    else
        echo "ZenML server is already running."
    fi
    
    # Check if MLflow server is already running
    if ! pgrep -f "$MLFLOW_PROCESS_PATTERN" > /dev/null; then
        echo "Starting MLflow server in $MLFLOW_DIR..."
        cmd="mlflow server \
        --backend-store-uri \"sqlite:///$MLFLOW_DIR/mlflow.db\" \
        --default-artifact-root \"$MLFLOW_DIR/mlruns\" \
        --host $MACHINE_IP \
        --port $MLFLOW_PORT \
        &> \"$MLFLOW_DIR/mlflow.log\" &"
        
        echo "$cmd"
        eval "$cmd"
        echo "MLflow server started."
    else
        echo "MLflow server is already running."
    fi
}

# Function to stop servers
stop_servers() {
    echo "Stopping ZenML and MLflow servers..."
    
    # Stop ZenML server
    ZENML_PID=$(pgrep -f "$ZENML_PROCESS_PATTERN")
    if [ -n "$ZENML_PID" ]; then
        echo "Stopping ZenML server (PID: $ZENML_PID)..."
        kill "$ZENML_PID"
        echo "ZenML server stopped."
    else
        echo "ZenML server is not running."
    fi
    
    # Stop MLflow server
    MLFLOW_PID=$(pgrep -f "$MLFLOW_PROCESS_PATTERN")
    if [ -n "$MLFLOW_PID" ]; then
        echo "Stopping MLflow server (PID: $MLFLOW_PID)..."
        pkill "$MLFLOW_PROCESS_PATTERN"
        echo "MLflow server stopped."
    else
        echo "MLflow server is not running."
    fi
}

# Parse command-line argument
if [ "$1" == "start" ]; then
    start_servers
    elif [ "$1" == "stop" ]; then
    stop_servers
else
    echo "Usage: $0 {start|stop}"
    exit 1
fi
