# Scene Graph Generation

This is pytorch implementation of our ECCV-2018 paper: [**Factorizable Net: An Efficient Subgraph-based Framework for Scene Graph Generation**](http://cvboy.com/publication/eccv2018_fnet/). <br> This project is based on our previous work: [**Multi-level Scene Description Network**](https://github.com/yikang-li/MSDN).


model|description
---|---
RPN (region proposal network)| a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position
Factorizable Net (MSDN based)|scene graph generation model with RPN

(reference github) https://github.com/yikang-li/FactorizableNet

---
1. [Progress](#Progress)
2. [Installation](#Installation)
3. [Dataset](#Dataset) 
4. [Training](#Training)
5. [Evaluate](#Evaluate)
6. [Inference](#Inference)
7. [Result](#Result)
8. [Comparison](#Comparison)
---
 
## Progress
- [x] Guide for Project Setup
- [x] Guide for Model Evaluation with pretrained model
- [x] Guide for Model Training
- [x] Uploading pretrained model and format-compatible datasets.


## Installation
### github (download)
~~~console
$ git clone https://github.com/pushdown99/fact.git
$ cd fact
~~~

### venv 

Read [INSTALL.md](INSTALL.md)

~~~console
python -m venv torch
source torch/bin/activate
python -m pip install --upgrade pip
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
#pip install colorama easydict opencv-python matplotlib
pip install pyyaml numpy==1.23.1 timer ipython matplotlib climage opencv-python==4.6.0.66 easydict sympy click Cython h5py
~~~

### build RPN library (using cython for c/c++ code)

~~~cosole
$ cd lib
$ make clean all
$ cd ..
~~~

### docker

for the runtime nvidia-docker environment installation <br>
read [NVIDIA-DOCKER-INSTALL](NVIDIA-DOCKER-INSTALL.md)
~~~console
$ ./docker.sh run
~~~
or 
~~~console
$ sudo docker run --gpus all -it --rm --runtime=nvidias pushdown99/fact bash
~~~

## jupyter notebook

~~~console
$ ./jupyter.sh
~~~
or
~~~console
$ jupyter notebook --ip=0.0.0.0 --port=8000 --NotebookApp.iopub_data_rate_limit=1.0e10
~~~
---

## Dataset
The model has been trained on train/val NIA dataset. You can download the dataset here. Note that test images are not required for this code to work.

dataset/train{prefix}.json
dataset/val{prefix}.json

dataset|prefix|ratio
---|---|---
normal||100%
fat|_fat|50%
small|_small|10%

Each element in the train.json file has such a structure :
"images/IMG_0061865_(...).jpg": ["relation1", "relation2", "relation3", "relation4", ..."relation10"], ...

In same way in the val.json :
"images/IMG_0061865_(...).jpg": ["relation1", "relation2", "relation3", "relation4", ..."relation10"], ...

##Dependencies
I have used the following versions for code work:

code work|version
---|---
python|3.8.10
torch|1.8.0+cu111
cuda|11.1
cudnn|8.0.5
numpy|1.23.1

#setting
For my training session, I have get best results with this options file :

model|option file
---|---
data| options/data.yaml
RPN|options/RPN/rpn.yaml
FN|options/models/msdn.yaml

## Evaluation
evaluate model with pretrained models

~~~console
python train_fn.py --evaluate --pretrained_model output/trained_models/model.h5
~~~

## Training
- Training Region Proposal Network (RPN). The **shared conv layers** are fixed. We also provide pretrained RPN and MSDN model on [nia](https://drive.google.com/drive/folders/1bdxKKJ9-53b7-Qp9ykX89zmKk55vSdAd?usp=sharing)
	
	~~~console
	# Train RPN 
	$ source torch/bin/activate
	$ python train_rpn.py --dataset_option=normal 
	~~~

- Training Factorizable Net (MSDN based): detailed training options are included in ```options/models/```.

	~~~console
	# Train F-Net on NIA-MSDN:
	$ source torch/bin/activate
	$ python train_fn.py
	~~~
	
	```--rpn xxx.h5``` can be ignored in end-to-end training from pretrained **VGG16**. Sometime, unexpected and confusing errors appear. *Ignore it and restart to training.*
	
- For better results, we usually re-train the model with additional epochs by resuming the training from the checkpoint with ```--resume ckpt```:

	~~~console
	# Resume F-Net training on NIA-MSDN:
	$ python train_fn.py --resume ckpt --epochs 30
	~~~
