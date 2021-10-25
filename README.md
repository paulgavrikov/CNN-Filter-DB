# CNN-Filter-DB

**An Empirical Investigation of Model-to-Model Distribution Shifts in Trained Convolutional Filters**<br>
Paul Gavrikov, Janis Keuper

![Distribution shifts of trained 3x3 convolution filters](./assets/kl_combined.png)

Paper: <not yet available>

Abstract: *We present first empirical results from our ongoing investigation of distribution shifts in image data used for various computer vision tasks. Instead of analyzing the original training and test data, we propose to study shifts in the learned weights of trained models. In this work, we focus on the properties of the distributions of dominantly used 3x3 convolution filter kernels. We collected and publicly provide a data set with over half a billion filters from hundreds of trained CNNs, using a wide range of data sets, architectures, and vision tasks. Our analysis shows interesting distribution shifts (or the lack thereof) between trained filters along different axes of meta-parameters, like data type, task, architecture, or layer depth. We argue, that the observed properties are a valuable source for further investigation into a better understanding of the impact of shifts in the input data to the generalization abilities of CNN models and novel methods for more robust transfer-learning in this domain.*

## Versions 
  
| Number | Changes |
|:---:|:---|
| v1.0 | Initial dataset as presented in the NeurIPS 2021 DistShift Workshop|

## Environment 
We have executed this with `Python 3.8.8` on `Linux 3.10.0-1160.24.1.el7.x86_64`. The scripts should however work with most python3 versions and OS.

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
Download `dataset.h5` from https://bit.ly/2Zc4xry. This file contains the filters and meta information as individual datasets. 
 
The filters are linked as a `Nx9` `numpy.float32` array under the `/filter` dataset. Every row is one filter and the row number is also the filter ID (i.e. the first row is filter ID 0). To reshape a filter `f` back to their original shape use `f.reshape(3, 3)`.
  
The meta information is stored as a `pandas.DataFrame` under `/meta`.

## Run
Adjust `dataset_path` in https://github.com/paulgavrikov/CNN-Filter-DB/blob/main/main.ipynb and run the cells.

## Citation 

If you find our work useful in your research, please consider citing:

```
<not yet available>
```
