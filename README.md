# MTWFNet
A New Method for Multimodal Micro-Expression Recognition

### requirements
certifi==2023.7.22
charset-normalizer==3.2.0
cmake==3.27.4.1
contourpy==1.1.0
cycler==0.11.0
dlib==19.24.2
filelock==3.12.3
fonttools==4.42.1
idna==3.4
importlib-resources==6.0.1
Jinja2==3.1.2
kiwisolver==1.4.5
lit==17.0.0rc4
MarkupSafe==2.1.3
matplotlib==3.7.2
mpmath==1.3.0
networkx==3.1
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-cupti-cu11==11.7.101
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
nvidia-cudnn-cu11==8.5.0.96
opencv-contrib-python==4.4.0.44
packaging==23.1
pandas==2.0.3
Pillow==10.0.0
pyparsing==3.0.9
python-dateutil==2.8.2
pytz==2023.3.post1
requests==2.31.0
scikit-learn @ file:///croot/scikit-learn_1690978916802/work
six==1.16.0
sympy==1.12
torch==2.0.1
torchaudio==2.0.2
torchstat==0.0.7
torchvision==0.15.2
tqdm==4.66.1
triton==2.0.0
typing_extensions==4.7.1
tzdata==2023.3
urllib3==2.0.4
xlrd==2.0.1


Additional notes
For other environment requirements, please see the requirements.txt document under the path

### Dataset Preparation

Firstly,calculate u,v,os,maguv using preprocess.py

The dataset and txt file format are as follows:
*sub01
   **sub01/EP02_01f/u.jpg,sub01/EP02_01f/v.jpg,sub01/EP02_01f/os.jpg,sub01/EP02_01f/maguv.jpg,1

*sub02
   **……

……

###Best Model Parameters

Saved in ./result/modelpth


### Training and Testing the MTWFNet

Training the 4-class data by using the following command:
```Bash  
cd “Code Path”
nohup python -u train_split.py > ./trainresult4C.log 2>&1 &
```

### Testing the best model for MTWFNet

Testing the 4-class data by using the following command:
```Bash  
cd “Code Path”
nohup python -u train_split.py > ./trainresult4C.log 2>&1 &
```




