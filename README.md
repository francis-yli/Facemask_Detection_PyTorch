# Facemask-Detection_PyTorch
A practice on applying object detection neural network for mask detection.

An implementation following [Jad Haddad's post](https://towardsdatascience.com/how-i-built-a-face-mask-detector-for-covid-19-using-pytorch-lightning-67eb3752fd61) on the same topic.

## Usage

### Training
```
python -m train_model
```

### Detecting in video
```
python -m detect_face_video [model path] [video path]
```

## Dataset
The dataset I used for training was from [RWFD](Real-World-Masked-Face-Dataset) Dataset.

Another way to implement this task could be using dataset of faces and dataset of faces with photoshopped masks. I saw several posts using such dataset have good results.
