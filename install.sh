#!/bin/bash

set -e

if ! which nvcc > /dev/null; then
  echo "ERROR: nvcc not found"
  exit
fi

curl https://pyenv.run | bash
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 2.7.17
pyenv virtualenv 2.7.17 yolo_ros
pyenv activate yolo_ros
python -m pip install -r requirements.txt
pyenv deactivate
