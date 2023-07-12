# Transactions business performance
 Evaluating small enterprises' business performance based on transactions in the supply chain

![](https://shields.io/badge/dependencies-Python_3.11-blue)
![](https://shields.io/badge/OS-Ubuntu_20.04-lightgrey)

## Acknowledgement

`stumpy`

This package is modified from https://github.com/TDAmeritrade/stumpy/tree/main/stumpy 

`deep_forest-0.1.7-cp311-cp311-linux_x86_64.whl`

This package is built from https://github.com/LAMDA-NJU/Deep-Forest with wheel. (Read the license also via this link.) It is only compatible in Linux system and Python 3.11.

If you're using other Python version, you have to build a wheel package. Building wheel package is not supported in virtual environment. We have to install Python 3.11 globally (if not installed previously) and set it as the default Python global environment. It is settled down if running the following commands.

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11
cd /usr/bin
sudo rm python3
ln -s python3.11 python3
sudo apt install python3-pip
cd ~
git clone https://github.com/LAMDA-NJU/Deep-Forest.git
cd ~/Deep-Forest/
sudo apt-get install python3.11-dev build-essential
python3.11 -m pip wheel .
```

To change the default Python global environment back: assume the original Python version is 3.10, run the following commands.

```bash
cd /usr/bin
sudo rm /usr/bin/python3
ln -s python3.10 python3
```

If you're using anaconda/miniconda, the behavior of Python version management is not tested.

This package requires the citation below.

```
@article{zhou2019deep,
  title={Deep forest},
  author={Zhou, Zhi-Hua and Feng, Ji},
  journal={National science review},
  volume={6},
  number={1},
  pages={74--86},
  year={2019},
  publisher={Oxford University Press}
}
```

