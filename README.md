# Facemask-Detection_PyTorch
A practice on applying object detection neural network for mask detection.

## Usage

### Training
```
python -m train_model
```

### Detecting in video
```
python -m detect_face_video [model path] [video path]
```

## Video example

![](facemask_test_video.mov)

## Dataset
The dataset I used for training was from [RWFD](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset) Dataset.

Another way to implement this task could be using dataset of faces and dataset of faces with photoshopped masks. I saw several posts using such dataset have good results.

## Reference

-[Face Mask Detection](https://github.com/chandrikadeb7/Face-Mask-Detection)

-[COVID-19: Face Mask Detector with OpenCV, Keras/TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)

-[How I built a Face Mask Detector for COVID-19 using PyTorch Lightning](https://towardsdatascience.com/how-i-built-a-face-mask-detector-for-covid-19-using-pytorch-lightning-67eb3752fd61)
