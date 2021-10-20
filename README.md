# CNN-Filter-DB

**An Empirical Investigation of Model-to-Model Distribution Shifts in Trained Convolutional Filters**<br>
Paul Gavrikov, Janis Keuper

![Distribution shifts of trained 3x3 convolution filters](./assets/kl_combined.png)

Paper:

Abstract: *We present first empirical results from our ongoing investigation of distribution shifts in image data used for various computer vision tasks. Instead of analyzing the original training and test data, we propose to study shifts in the learned weights of trained models. In this work, we focus on the properties of the distributions of dominantly used 3x3 convolution filter kernels. We collected and publicly provide a data set with over half a billion filters from hundreds of trained CNNs, using a wide range of data sets, architectures, and vision tasks. Our analysis shows interesting distribution shifts (or the lack thereof) between trained filters along different axes of meta-parameters, like data type, task, architecture, or layer depth. We argue, that the observed properties are a valuable source for further investigation into a better understanding of the impact of shifts in the input data to the generalization abilities of CNN models and novel methods for more robust transfer-learning in this domain.*


## Environment 
Python 3.8.8

Linux 3.10.0-1160.24.1.el7.x86_64

To install all necessary modules please run:
```
pip install -r requirements.txt
```

or install these modules manually with your desired package manager:
```
numpy==1.21.2
scipy
scikit-learn==0.24.1
matplotlib==3.4.1
pandas==1.1.4
fast-histogram==0.10
KDEpy==1.1.0
tqdm==4.53.0
colorcet==2.0.6
h5py==3.1.0
```


## Prepare 
Download dataset.h5 from https://drive.google.com/drive/folders/11ODURpMXY-LWFuu46rlUE69MDDS-48I0. This file contains the filters and meta information.

## Run
Adjust `dataset_path` in https://github.com/paulgavrikov/CNN-Filter-DB/blob/main/main.ipynb and run the cells.

## Citation 

If you find our work useful in your research, please consider citing:

```
```
