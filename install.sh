#/bin/bash
echo "Set up virtualenv for scanimate..."
virtualenv -p python3.6 ./venv/scanimate
. ./venv/scanimate/bin/activate

echo "Installing torch..."
echo "If you are using other versions (default python3.6 cuda 10.1), change the cuda version and python version in ./install.sh"
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

echo "Installing pytorch3d..."
echo "If you are using other versions (default python3.6 cuda 10.1), change the cuda version and python version in ./install.sh"
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py36_cu101_pyt171/download.html

echo "Installing other dependencies..."
pip install -r requirements.txt

echo "Installing customized smpl code"
cd smpl
python3 setup.py install
cd ../

echo "Done!"