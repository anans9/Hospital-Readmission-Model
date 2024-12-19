#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check the operating system
is_mac() {
    [[ "$OSTYPE" == "darwin"* ]]
}

is_windows() {
    [[ "$OS" == "Windows_NT" ]]
}

is_ubuntu() {
    [[ "$(uname -a)" == *"Ubuntu"* ]]
}

# Logic for macOS
if is_mac; then
    echo "Detected macOS"

    # Install Homebrew
    if ! command_exists brew; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
        echo "Homebrew is already installed."
    fi

    # Install Python
    if ! command_exists python3; then
        echo "Python3 not found. Installing Python3..."
        brew install python
    else
        echo "Python3 is already installed."
    fi

    # Create a virtual environment
    if [ ! -d "env" ]; then
        echo "Creating a virtual environment..."
        python3 -m venv env
    else
        echo "Virtual environment already exists."
    fi

    # Activate the virtual environment
    echo "Activating the virtual environment..."
    source env/bin/activate

    # Upgrade pip and setuptools
    echo "Upgrading pip and setuptools..."
    pip install --upgrade pip setuptools

    # Install required Python packages
    echo "Installing required packages..."
    pip install -r requirements.txt
    brew install libomp

    echo "Setup complete. Running the Python script..."

    # Run the python script
    python3 run.py "Initial Diabetics Data 10000.csv"

    # Deactivate the virtual environment
    echo "Deactivating the virtual environment..."
    deactivate

# Logic for Windows
elif is_windows; then
    echo "Detected Windows"

    # Check if Python is installed
    if ! command_exists python; then
        echo "Python not found. Please install Python 3 from https://www.python.org/downloads/ and rerun this script."
        exit 1
    else
        echo "Python is already installed."
    fi

    # Create a virtual environment
    if [ ! -d "env" ]; then
        echo "Creating a virtual environment..."
        python -m venv env
    else
        echo "Virtual environment already exists."
    fi

    # Activate the virtual environment
    echo "Activating the virtual environment..."
    source env/Scripts/activate

    # Upgrade pip and setuptools
    echo "Upgrading pip and setuptools..."
    pip install --upgrade pip setuptools

    # Install required Python packages
    echo "Installing required packages..."
    pip install -r requirements.txt

    echo "Setup complete. Running the Python script..."

    # Run the python script
    python run.py "Initial Diabetics Data 10000.csv"

    # Deactivate the virtual environment
    echo "Deactivating the virtual environment..."
    deactivate

# Logic for Ubuntu Linux
elif is_ubuntu; then
    echo "Detected Ubuntu Linux"

    # Update package list and install Python if not installed
    if ! command_exists python3; then
        echo "Python3 not found. Installing Python3..."
        sudo apt-get update
        sudo apt-get install -y python3 python3-venv python3-pip
    else
        echo "Python3 is already installed."
    fi

    # Create a virtual environment
    if [ ! -d "env" ]; then
        echo "Creating a virtual environment..."
        python3 -m venv env
    else
        echo "Virtual environment already exists."
    fi

    # Activate the virtual environment
    echo "Activating the virtual environment..."
    source env/bin/activate

    # Upgrade pip and setuptools
    echo "Upgrading pip and setuptools..."
    pip install --upgrade pip setuptools

    # Install required Python packages
    echo "Installing required packages..."
    pip install -r requirements.txt

    echo "Setup complete. Running the Python script..."

    # Run the python script
    python3 run.py "Initial Diabetics Data 10000.csv"

    # Deactivate the virtual environment
    echo "Deactivating the virtual environment..."
    deactivate

# Unsupported OS
else
    echo "Unsupported operating system. This script only supports macOS, Windows, and Ubuntu Linux."
    exit 1
fi
