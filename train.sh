# activate python environment
source venv/bin/activate

# set cuda_visible_devices
export CUDA_VISIBLE_DEVICES=0,1

# run training script
PYTHONPATH=$(pwd):$PYTHONPATH python ./src/train.py