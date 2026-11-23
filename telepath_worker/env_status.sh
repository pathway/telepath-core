set -x

python --version

python -c "import torch; print('pytorch',torch.__version__)"
python -c "import torch; print(torch.cuda.is_available()); "
python -c "import torch; print(torch.cuda.get_device_name(0))"


cat /etc/issue
uname -a
free -h
df -h
ifconfig

cat /usr/local/cuda/version.txt
nvcc --version
nvidia-smi

