**********************
CUDA TORCH
**********************
python -m venv torch
source torch/bin/activate
python -m pip install --upgrade pip
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
#pip install colorama easydict opencv-python matplotlib
pip install pyyaml numpy==1.23.1 timer ipython matplotlib climage opencv-python==4.6.0.66 easydict sympy click Cython h5py
python -m ipykernel install --user --name torch --display-name "torch"

