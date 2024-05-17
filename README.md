# FSLFDE

This is the repository of the paper "__A Framework for Learning Depth From a Flexible Subset of Dense and Sparse Light Field Views__"  (__TIP 2019__).

By [Jinglei Shi](https://jingleishi.github.io/),  [Xiaoran Jiang](https://scholar.google.com/citations?hl=zh-CN&user=zvdY0EcAAAAJ&view_op=list_works&sortby=pubdate)  and  [Christine Guillemot](https://people.rennes.inria.fr/Christine.Guillemot/)

<[Project page](http://clim.inria.fr/research/FlexDepthEstim/index.html)>,   <[Paper link](https://ieeexplore.ieee.org/document/8743559)>

## Dependencies
```
python==xxx
tensorflow==xxx
```

## Evaluation
```
TODO Later
```

## Datasets
We have created "__INRIA Synthetic Light Field Datasets__" for various light field processing tasks: __Dense Light Field Dataset__ ([__DLFD__](https://pan.baidu.com/s/1tywF8hcgx4i5IDRQKIEV_A), captcha "__dlfd__") and __Sparse Light Field Dataset__ ([__SLFD__](https://pan.baidu.com/s/1jzFkTfJyx2XhkpF6nItoBQ), captcha "__slfd__") in __.zip__, each light field within the dataset has angular resolution $9 \times 9$ and spatial resolution $512 \times 512$. We offer in each scene folder all sub-aperture images in __.PNG__, disparity maps in both __.npy__ and __.mat__, and a __.cfg__ file containing camera parameters.


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
