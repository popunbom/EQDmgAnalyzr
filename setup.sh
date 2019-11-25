#!/bin/bash

############################
######  Setup Script  ######
############################

PYTHON_3=""

# Check Python 3.x
if type "python3" > /dev/null 2>&1; then
  if [[ $(python -c "from sys import version_info as vi; print(vi.minor >= 6)") == "True" ]]; then
    PYTHON_3="python3"
  fi
fi
if type "python" > /dev/null 2>&1; then
  if [[ $(python -c "from sys import version_info as vi; print(vi.major == 3 and vi.minor >= 6)") == "True" ]]; then
    PYTHON_3="python"
  fi
fi

if [[ $PYTHON_3 == "" ]]; then
  echo "Python >3.6 is not available -- aborted."
  exit -1
fi

# Create venv
if [[ ! -e "${PWD}/venv" ]]; then
  echo "Create venv ... "
  ${PYTHON_3} -m venv ./venv
  echo "  -> Successed to create ${PWD}/venv"
fi

# Activate
ACTIVATE_SCRIPT=""
if [[ $(basename $SHELL) == "csh" ]]; then
  ACTIVATE_SCRIPT="./venv/bin/activate.csh"
elif [[ $(basename $SHELL) == "fish" ]]; then
  ACTIVATE_SCRIPT="./venv/bin/activate.fish"
else
  ACTIVATE_SCRIPT="./venv/bin/activate"
fi

source ${ACTIVATE_SCRIPT}

# Upgrade pip
pip install --upgrade pip

# Install packages
pip install -r ./requirements.txt

echo "Success to Setup !"
echo "To activete venv, run 'source ${ACTIVATE_SCRIPT}'"
