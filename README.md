## LeNet-5-nearly-from-scratch
This repository represents LeNet-5 on digit handwritting dataset MNIST.  
It is built from the DataLoader to load MNIST Dataset to the main.py. Additionally, it also contains collate_function and more...   
## Requirement
`python 3.10.5` 
`idx2numpy` 
`numpy` 
`pytorch`
`sklearn`  
`tqdm`   
## How to run this repository  
Firstly, you have to clone this repo by using `git clone` command.  
Then, activating virtual environment and install `Requirenment packet`.  
Choose the right directory to the corresponding (`train`, `dev`, `test` set).  
Run  
## How to access `NVIDIA` GPU on your PC  
In command line, type nvidia-smi. Then you have to look for `CUDA Version`.   
Browse to `pytorch.org` and choose `CUDA version` that lower than yours.   
Copy its command and run it in your terminal, you now have the ability to access to your `NVIDIA` GPU.  
