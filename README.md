# CNN Filter DB: An Empirical Investigation of Trained Convolutional Filters (CVPR2022 ORAL)
Paul Gavrikov, Janis Keuper

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]


[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

[Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Gavrikov_CNN_Filter_DB_An_Empirical_Investigation_of_Trained_Convolutional_Filters_CVPR_2022_paper.html) | [ArXiv](https://arxiv.org/abs/2203.15331) | [HQ Poster](https://zenodo.org/record/6687455#.YrOAZi9w2MI) | [Oral Presentation](https://ieeecs.warpwire.com/w/l5sAAA/)

Abstract: *Currently, many theoretical as well as practically relevant questions towards the transferability and robustness of Convolutional Neural Networks (CNNs) remain unsolved. While ongoing research efforts are engaging these problems from various angles, in most computer vision related cases these approaches can be generalized to investigations of the effects of distribution shifts in image data.
In this context, we propose to study the shifts in the learned weights of trained CNN models. Here we focus on the properties of the distributions of dominantly used 3x3 convolution filter kernels. We collected and publicly provide a data set with over 1.4 billion filters from hundreds of trained CNNs, using a wide range of data sets, architectures, and vision tasks.
In a first use case of the proposed data set, we can show highly relevant properties of many publicly available pre-trained models for practical applications: I) We analyze distribution shifts (or the lack thereof) between trained filters along different axes of meta-parameters, like visual category of the data set, task, architecture, or layer depth. Based on these results, we conclude that model pre-training can succeed on arbitrary data sets if they meet size and variance conditions. II) We show that many pre-trained models contain degenerated filters which make them less robust and less suitable for fine-tuning on target applications.*

![Poster](./assets/cvpr22_poster_cnnfilterdb_min.png)

> **⚠ Note**  
> We have received interest of many individuals in using the metrics we propose in our paper to better understand CNN training. Therefore, we have released a torch library here: https://github.com/paulgavrikov/torchconvquality

## Versions 
  
| Version | Access | Changes |
|:---:|:---:|:---|
| v1.0.0 **(latest)** | https://doi.org/10.5281/zenodo.6371680 | Dataset as presented at CVPR 2022|
| v0.1.0 | https://kaggle.com/paulgavrikov/cnn-filter-db | Initial dataset as presented in the NeurIPS 2021 DistShift Workshop. Workshop Paper: https://openreview.net/forum?id=2st0AzxC3mh Workshop Poster: https://doi.org/10.5281/zenodo.6392142|

If you are looking for our specialized dataset on Robustness head to https://github.com/paulgavrikov/cvpr22w_RobustnessThroughTheLens.

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
tables==3.6.1
```


## Prepare 
Download `dataset.h5`. This file contains the filters and meta information as individual datasets. 
If the filename ends with `.xz` you first need to decompress it: `xz -dv data.csv.xz`. Note that this will increase size by 225%.

The filters are linked as a `Nx9` `numpy.float32` array under the `/filter` dataset. Every row is one filter and the row number is also the filter ID (i.e. the first row is filter ID 0). To reshape a filter `f` back to its original shape use `f.reshape(3, 3)`.
  
The meta information is stored as a `pandas.DataFrame` under `/meta`. Following is an *out of order* list of column keys with a short description. Other column keys can and should be ignored. The table has a Multiindex on `[model_id, conv_depth, conv_depth]`.
  
| Column                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Description   |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------|
| model_id                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Unique int ID of the model. |
| conv_depth                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Convolution depth of the extracted filter i.e. how many convolution layers were hierarchically below the layer this filter was extracted from.  |
| conv_depth_norm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Similar to `conv_depth` but normalized by the maximum `conv_depth`. Will be a flaot betwenn 0 (first layers) .. 1 (towards head). |
| filter_ids                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | List of Filter IDs that belong to this record. These can directly be mapped to the rows of the filter array. |
| model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Unique string ID of the model. Typically, but not reliably in the format {name}_{trainingset}_{onnx opset}. |
| producer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Producer of the ONNX export. Typically various versions of PyTorch. |
| op_set                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Version of the ONNX operator set used for export. |
| depth                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Total hierarchical depth of the model including all layers. |
| Name                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Name of the model. Not necessarily unique. |
| Paper                                                                                                                                                       |  Link to the Paper. Not always populated. |  
| Pretraining-Dataset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Name of the pretraining dataset(s) if pretrained. Multiple datr sets are seperated by commas. |
| Training-Dataset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Name of the training dataset(s). Multiple datr sets are seperated by commas.|
| Datatype                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Visual, manual categorization of the training datatsets. |
| Task                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Task of the model. |
| Accessible                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Represents where the model can be found. Typically this is a link to GitHub.  |
| Dataset URL                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | URL of the training dataset. Usually only entered for exotic datasets. |
| total_filters                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Total number of convolution filters in this model. |
| 3x3_filter_share                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | The share of 3x3 filters compared to all other conv filters. |
| (X, Y) filters                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Represents how often filters of shape `(X, Y)` were found in the source model. |
| Conv, Add, Relu, MaxPool, Reshape, MatMul, Transpose, BatchNormalization, Concat, Shape, Gather, Softmax, Slice, Unsqueeze, Mul, Exp, Sub, Div, Pad, InstanceNormalization, Upsample, Cast, Floor, Clip, ReduceMean, LeakyRelu, ConvTranspose, Tanh, GlobalAveragePool, Gemm, ConstantOfShape, Flatten, Squeeze, Less, Loop, Split, Min, Tile, Sigmoid, NonMaxSuppression, TopK, ReduceMin, AveragePool, Dropout, Where, Equal, Expand, Pow, Sqrt, Erf, Neg, Resize, LRN, LogSoftmax, Identity, Ceil, Round, Elu, Log, Range, GatherElements, ScatterND, RandomNormalLike, PRelu, Sum, ReduceSum, NonZero, Not | Represents how often this ONNX operator was found in the original model. Please note that individual operators may have been fused in later ONNX opsets. |


## Run
Adjust `dataset_path` in https://github.com/paulgavrikov/CNN-Filter-DB/blob/main/main.ipynb and run the cells.

## Citation 

If you find our work useful in your research, please consider citing:

```
@InProceedings{Gavrikov_2022_CVPR,
    author    = {Gavrikov, Paul and Keuper, Janis},
    title     = {CNN Filter DB: An Empirical Investigation of Trained Convolutional Filters},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {19066-19076}
}
```

and the dataset:
```
@dataset{cnnfilterdb2022,
  author       = {Paul Gavrikov and
                  Janis Keuper},
  title        = {CNN-Filter-DB},
  month        = jun,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.6371680},
  url          = {https://doi.org/10.5281/zenodo.6371680}
}
```
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6371680.svg)](https://doi.org/10.5281/zenodo.6371680) 
### Legal
This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

Funded by the Ministry for Science, Research and Arts, Baden-Wuerttemberg, Germany Grant 32-7545.20/45/1 (Q-AMeLiA).
