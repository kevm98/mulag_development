#!/usr/bin/env bash

# Copyright (c) 2022-2024, The RRLab Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

# Get source directory
export RRLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

#==
# Helper functions
#==

# Extract Python executable from the environment
extract_python_exe() {
    # Check if using conda
    if ! [[ -z "${CONDA_PREFIX}" ]]; then
        # Use conda python
        local python_exe=${CONDA_PREFIX}/bin/python
    else
        # Fallback to system Python
        local python_exe=$(which python3)
    fi

    # Check if the Python path exists
    if [ ! -f "${python_exe}" ]; then
        echo -e "[ERROR] Unable to find Python executable at: '${python_exe}'" >&2
        exit 1
    fi

    # Return the result
    echo ${python_exe}
}

# Check if the input directory is a Python extension and install the module
install_rrlab_extension() {
    # Retrieve the Python executable
    python_exe=$(extract_python_exe)

    # If the directory contains setup.py, then install the Python module
    if [ -f "$1/setup.py" ]; then
        echo -e "\tInstalling module: $1"
        ${python_exe} -m pip install --no-build-isolation --editable $1 # without --no-build-isolation can lead to toml import error
    fi
}

#==
# Main
#==

# Check argument provided
if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    echo -e "\nUsage: rrlab.sh [-i] [-p] -- Utility to manage RRLab."
    echo -e "\nOptional arguments:"
    echo -e "\t-i, --install         Install extensions in the RRLab repository."
    echo -e "\t-p, --python          Run the Python executable."
    exit 1
fi

# Pass the arguments
while [[ $# -gt 0 ]]; do
    # Read the key
    case "$1" in
        -i|--install)
            # Install the Python packages in the RRLab directory
            echo "[INFO] Installing extensions in the RRLab repository..."
            export -f extract_python_exe
            export -f install_rrlab_extension
            # Source directory for extensions
            find -L "${RRLAB_PATH}/extensions" -mindepth 1 -maxdepth 1 -type d -exec bash -c 'install_rrlab_extension "{}"' \;
            shift # Past argument
            ;;
        -p|--python)
            # Run the Python executable
            python_exe=$(extract_python_exe)
            echo "[INFO] Using Python from: ${python_exe}"
            shift # Past argument
            ${python_exe} "$@"
            break
            ;;
        *)
            echo "[Error] Invalid argument provided: $1"
            echo -e "\nUsage: rrlab.sh [-i] [-p] -- Utility to manage RRLab."
            exit 1
            ;;
    esac
done
