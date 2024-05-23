# FSLFDE

This is the repository of the paper "__A Framework for Learning Depth From a Flexible Subset of Dense and Sparse Light Field Views__"  (__TIP 2019__).

By [Jinglei Shi](https://jingleishi.github.io/),  [Xiaoran Jiang](https://scholar.google.com/citations?hl=zh-CN&user=zvdY0EcAAAAJ&view_op=list_works&sortby=pubdate)  and  [Christine Guillemot](https://people.rennes.inria.fr/Christine.Guillemot/)

<[Project page](http://clim.inria.fr/research/FlexDepthEstim/index.html)>,   <[Paper link](https://ieeexplore.ieee.org/document/8743559)>

## Dependencies
```
python==2.X or <=3.6
tensorflow==1.2.1
cuda version==8.0.27 compatible GPU (tested on NVIDIA Tesla P-100)
```

## Contents
Folder '**fn2**': the code of FlowNet 2.0 and some tool functions.

Folder '**models**': two well trained models (one for densely sampled light fields, the other for sparsely sampled light fields). 

Download links: [dense model](https://pan.baidu.com/s/13beodQnn7PnAgB-Mrz82Pw?pwd=0000)  &  [sparse model](https://pan.baidu.com/s/1ngwujIxeUPknGSMAmk7R6w?pwd=0001)

**refinement.py**: the code for the refinement network.

**warper.py**: it inclueds functions that warp images towards desired position.

**pipeline.py**: our proposed pipeline, which integrates both FlowNet 2.0 and refinement network together.

**test.py**: 




## Datasets
We have created the "__INRIA Synthetic Light Field Datasets__" tailored for diverse light field processing tasks: __Dense Light Field Dataset__ ([__DLFD__](https://pan.baidu.com/s/1tywF8hcgx4i5IDRQKIEV_A), captcha "__dlfd__") and __Sparse Light Field Dataset__ ([__SLFD__](https://pan.baidu.com/s/1jzFkTfJyx2XhkpF6nItoBQ), captcha "__slfd__"), each dataset in a .zip format. Every light field included in the datasets boasts an angular resolution of $9 \times 9$ and a spatial resolution $512 \times 512$. Within each scene folder, we provide all sub-aperture images in __.PNG__ format, alongside disparity maps available in both __.npy__ and __.mat__ file formats, and a __.cfg__ file containing camera parameters.


## Citation
Please consider citing our work if you find it useful.
```
@article{shi2019depth,
    title={A Framework for Learning Depth From a Flexible Subset of Dense and Sparse Light Field Views},
    author={Jinglei Shi and Xiaoran Jiang and Christine Guillemot},
    journal=TIP,
    volume={28},
    number={12},
    pages={5867-5880},
    month={Dec},
    year={2019}}
```
