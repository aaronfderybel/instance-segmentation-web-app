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
