# ML-VRPose

三点トラッキングのVR機器にカメラを一台設置することでフルボディトラッキングを行う


## Requirement

事前にインストールしておく必要のあるもの
- SteamVR

必要なライブラリ
- mediapipe
- opencv-python
- pyopenvr
- numpy
- matplotlib


## Installation

仮想環境の構築
```
pip install --upgrade pip
pip install --upgrade setuptools
py -m venv workspace
```

ライブラリ
```
pip install --upgrade mediapipe opencv-python openvr numpy matplotlib
```


## Usage

仮想環境の有効化
```
workspace\Scripts\activate
```

仮想環境の無効化
```
deactivate
```


## Note

- [ToDo](https://github.com/Mokuichi147/ML-VRPose/projects/1)