

# <center>第三次实验报告</center>

<center> SA24001065 余家玮</center>

## Requirements

安装程序所需环境请执行下面语句

```python
pip install -r requirements.txt
pip install dlib
```

安装draggan请参考/draggan/readme.md

将draggan中对应viz目录下的函数进行替换即可

使用dlib前需要到该网站下载模型http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

演示视频直接下载演示视频.MP4观看，无法直接在md文件中加载

## 实验原理

1. GAN实现pix2pix，相比上次的FCN网络多了一个判别器，主要目的是将生成图片和语义图片或者真值图片进行组合与真值图片进行对比给出误差。并且在l1误差的基础上加上gan误差项。

2. 利用draggan实现自动图像编辑，利用dlib人脸识别将所需要的特征点标定好变换即可。无须手动指定原始点和目标点。

## 实验结果

运行下面语句可以进行pix2pix由语义图片生成真实图片

```python
python train.py
```

运行下面语句可以进行draggan

```api
.\scripts\gui.bat
```


以下是语义图像生成和draggan自动编辑（笑容）的实验结果

<table>
     <tr>
        <td><center><img src=train_results\epoch_515\result_1.png height="150%"></center></td>
    	<td><center><img src=train_results\epoch_515\result_2.png height="150%"> </center></td>
        <td><center><img src=train_results\epoch_515\result_3.png height="150%"> </center></td>
    </tr>
    <tr>
    <td><center><img src=train_results\epoch_515\result_5.png height="150%"> </center></td>
    	<td><center><img src=train_results\epoch_515\result_4.png height="150%"></center></td>
    </tr>

<video width="320" height="240" controls>
    <source src="演示视频.mp4" type="video/mp4">
</video>
