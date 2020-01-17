# Humanflow2
This is an official repository

*Anurag Ranjan, David T. Hoffmann, Dimitrios Tzionas, Siyu Tang, Javier Romero, and Michael J. Black.* Learning Multi-Human Optical Flow. IJCV 2019.


[[Project Page]](https://humanflow.is.tue.mpg.de/)
[[Arxiv]](https://arxiv.org/abs/1910.11667)

## Prerequisites
Download Multi-Human Optical Flow dataset from [here](https://humanflow.is.tue.mpg.de).

Download pre-trained PWC-Net models from [NVlabs/PWC-Net](https://github.com/NVlabs/PWC-Net) and store them in `models/` directory.

Install Pytorch. Install dependencies using
```sh
pip3 install -r requirements.txt
```

If there are issues with the correlation module, compile it from source - [ClementPinard/Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension).
## Training
For finetuning SPyNet on Multi-Human Optical Flow dataset use:
```sh
python main.py PATH_TO_DATASET --dataset humanflow -a spynet --div-flow 1 -b8 -j8 --lr LEARNING_RATE -w 1.0 1.0 1.0 1.0 1.0 --name NAME_OF_EXPERIMENT
```

For finetuning PWC-Net on Multi-Human Optical Flow dataset use:

```sh
python main.py PATH_TO_DATASET --dataset humanflow -a pwc --div-flow 20 -b8 -j8 --lr LEARNING_RATE --name NAME_OF_EXPERIMENT
```
## Testing
To test SPyNet trained on Multi-Human Optical Flow dataset, use
```sh
python test_humanflow.py PATH_TO_DATASET --dataset humanflow --arch spynet --div-flow 1 --pretrained pretrained/spynet_MHOF.pth.tar
```

To test PWC-Net trained on Multi-Human Optical Flow dataset, use
```sh
python test_humanflow.py PATH_TO_DATASET --dataset humanflow --arch pwc --div-flow 20 --no-norm  --pretrained pretrained/pwc_MHOF.pth.tar
```
## Acknowledgements
We thank Clement Pinard for his github repository [ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch). We use it as our code base. PWCNet is taken from [NVlabs/PWC-Net](https://github.com/NVlabs/PWC-Net). SPyNet implementation is taken from [sniklaus/pytorch-spynet](https://github.com/sniklaus/pytorch-spynet). Correlation module is taken from [ClementPinard/Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension).
