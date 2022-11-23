# Instance segmentation web app with fast-api and yolov7

This project contains a simple local run web-app to be able to perform instance segmentation on pictures you upload using the yolov7 algorithm. The yolov7 algorithm was pretrained on the coco dataset and supports all the standard coco-classes, see: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda 

The web-app was built using lightweigth fastapi library.
![interface](assets/example.PNG)


## Installation

Recommended to create a virtual environment, once activate install cython with the command

```bash
pip install cython
```

Install pytorch using a command from terminal suitable for your hardware and OS.
https://pytorch.org/get-started/locally/

Go to the detectron2 folder and install required packages
```bash
cd detectron2
pip install -e .
```

Install remaining packages
```bash
pip install -r requirements.txt
```

Test installation
```bash
python
>> import detectron2
>> import torch
``` 

Note: depending on torch version you use you might encounter an error when performing upscaling.
Go to your locally installed torch library and open `torch/nn/modules/upsampling.py` remove or comment out the recompute_scale_factor argument.

![adapt source code](assets/error.PNG)

## Usage
Start the web app using uvicorn from the command line
```bash
cd yolov7_mask
uvicorn main:app --host 0.0.0.0 --port 8000
```

When running the web app from a remote machine it's advised to add --host 0.0.0.0 argument. You can acess the web ap remotely using the ip of your remote machine in the network (this is not 0.0.0.0 but the ip or dns name of your remote machine).

If you run this local the host argument can be omitted and your web app can be acessed from `http://local:8000`. You can adapt the port `8000` if needed.

## References
This project was based upon previous work.
* Yolov7 Instance segmentation tutorial of official github: https://github.com/WongKinYiu/yolov7/blob/main/tools/instance.ipynb
* FastAPI example https://github.com/bhimrazy/Image-Recognition-App-using-FastAPI-and-PyTorch

