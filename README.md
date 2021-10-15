# CNN-Filter-DB

**An Empirical Investigation of Model-to-Model Distribution Shifts in Trained Convolutional Filters**<br>
Paul Gavrikov, Janis Keuper

Paper:

Abstract: *We present first empirical results from our ongoing investigation of distribution shifts in image data used for various computer vision tasks. Instead of analyzing the original training and test data, we propose to study shifts in the learned weights of trained models. In this work, we focus on the properties of the distributions of dominantly used 3x3 convolution filter kernels. We collected and publicly provide a data set with over half a billion filters from hundreds of trained CNNs, using a wide range of data sets, architectures, and vision tasks. Our analysis shows interesting distribution shifts (or the lack thereof) between trained filters along different axes of meta-parameters, like data type, task, architecture, or layer depth. We argue, that the observed properties are a valuable source for further investigation into a better understanding of the impact of shifts in the input data to the generalization abilities of CNN models and novel methods for more robust transfer-learning in this domain.*


## Environment 
Python 3.8.8
Linux 3.10.0-1160.24.1.el7.x86_64
To install all necessary modules please run (or install manually with your desired package manager).
```
pip install -r requirements.txt
```