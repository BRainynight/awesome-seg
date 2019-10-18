# 簡介
Fix from [Tramac's repositories](https://github.com/Tramac/awesome-semantic-segmentation-pytorch).
因為是在Google Colab上訓練，擔心被切斷，所以增加了自動上傳到Google Drive的片段。  
# 我所做的調整: 
## Train 階段
1. \scripts\train.py :    
  * class Trainer(object):     
    在`self.criterion = get_segmentation_loss`的參數裡面，多傳入一個(nclass = xx)    
    在其他地方，nclass可以透過 nclass=datasets[dataset].NUM_CLASS 的方式取得，但是這需要調用到 from dataloader import datasets.    
    這個檔案對 train.py來說太遙遠，暫且用手動輸入帶過他。    
  * add function `save_to_Gdrive`:     
    我自己另外增加了上傳到colab的包，修改過的colab+pydrive也放在這個respository裡面。    
  
2. 

## Eval 階段
1. \core\data\dataloader\mydata.py:  
  基於cityscapes.py 修改而成，其中因為手上自己資料集的label檔案，雖是黑白png，卻是以三通到的方式儲存。
  書需要轉換成灰階圖，否則評價 (socre.py )分數上，會因為矩陣大小不符合而報錯。
  * `def __getitem__():  `
    `mask = Image.open(self.mask_paths[index]).convert('L')`
    
  ```
  File "/content/awesome-semantic-segmentation-pytorch/core/utils/score.py", line 76, in batch_pix_accuracy
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
  RuntimeError: The size of tensor a (640) must match the size of tensor b (3) at non-singleton dimension 3
```
    
    
    
使用自己的資料集時

## Demo 階段    

1. \scripts\demo.py:       
沒有將 arg 的整個參數傳送給get_model，添加 `**vars(args)`  。    
因為 get_model 內部使用到的是 kwargs (字典方式取值) ，如果只傳入args，**kwargs根本是空的**，所以使用上述方式傳入字典形式參數索引。    
```
model = get_model(args.model, **vars(args), pretrained=True, root=args.save_folder).to(device)
```

2. model_zoo.py:       
in function get_icnet_resnet50_citys :      
```    
    kwargs.pop("dataset")     
    net = _models[name](**kwargs)
    return net 
```
因為決定用哪個模型，是由最初的參數 --model 來決定。    
如果把 dataset 傳入 _models 的模型中，會產生 **dataset 被指定兩次**的狀況，所以在這裡先把它刪掉 (_models的模型自己會指定 dataset)。    
不能從一開始就少掉這個指令是因為 demo.py 裡面需要這個參數。    
```
mask = get_color_pallete(pred, args.dataset)
```

**PS.** `--dataset citys` 這個指令不知道為何，一直沒辦法輸入成功，我直接把預設改成citys。  


-------

# Semantic Segmentation on PyTorch
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]

This project aims at providing a concise, easy-to-use, modifiable reference implementation for semantic segmentation models using PyTorch.

<p align="center"><img width="100%" src="docs/weimar_000091_000019_gtFine_color.png" /></p>

## Installation
```
# semantic-segmentation-pytorch dependencies
pip install ninja tqdm

# follow PyTorch installation in https://pytorch.org/get-started/locally/
conda install pytorch torchvision -c pytorch

# install PyTorch Segmentation
git clone https://github.com/Tramac/awesome-semantic-segmentation-pytorch.git

# the following will install the lib with symbolic links, so that you can modify
# the files if you want and won't need to re-build it
cd awesome-semantic-segmentation-pytorch/core/nn
python setup.py build develop
```

## Usage
### Train
-----------------
- **Single GPU training**
```
# for example, train fcn32_vgg16_pascal_voc:
python train.py --model fcn32s --backbone vgg16 --dataset pascal_voc --lr 0.0001 --epochs 50
```
- **Multi-GPU training**

```
# for example, train fcn32_vgg16_pascal_voc with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --model fcn32s --backbone vgg16 --dataset pascal_voc --lr 0.0001 --epochs 50
```

### Evaluation
-----------------
- **Single GPU evaluating**
```
# for example, evaluate fcn32_vgg16_pascal_voc
python eval.py --model fcn32s --backbone vgg16 --dataset pascal_voc
```
- **Multi-GPU evaluating**
```
# for example, evaluate fcn32_vgg16_pascal_voc with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS eval.py --model fcn32s --backbone vgg16 --dataset pascal_voc
```
### Demo
```
cd ./scripts
python demo.py --model fcn32s_vgg16_voc --input-pic ./datasets/test.jpg
```

```
.{SEG_ROOT}
├── scripts
│   ├── demo.py
│   ├── eval.py
│   └── train.py
```

## Support

#### Model

- [FCN](https://arxiv.org/abs/1411.4038)
- [ENet](https://arxiv.org/pdf/1606.02147)
- [PSPNet](https://arxiv.org/pdf/1612.01105)
- [ICNet](https://arxiv.org/pdf/1704.08545)
- [DeepLabv3](https://arxiv.org/abs/1706.05587)
- [DeepLabv3+](https://arxiv.org/pdf/1802.02611)
- [DenseASPP](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)
- [EncNet](https://arxiv.org/abs/1803.08904v1)
- [BiSeNet](https://arxiv.org/abs/1808.00897)
- [PSANet](https://hszhao.github.io/papers/eccv18_psanet.pdf)
- [DANet](https://arxiv.org/pdf/1809.02983)
- [OCNet](https://arxiv.org/pdf/1809.00916)
- [CGNet](https://arxiv.org/pdf/1811.08201)
- [ESPNetv2](https://arxiv.org/abs/1811.11431)
- [CCNet](https://arxiv.org/pdf/1811.11721)
- [DUNet(DUpsampling)](https://arxiv.org/abs/1903.02120)
- [FastFCN(JPU)](https://arxiv.org/abs/1903.11816)
- [LEDNet](https://arxiv.org/abs/1905.02423)
- [Fast-SCNN](https://github.com/Tramac/Fast-SCNN-pytorch)
- [LightSeg](https://github.com/Tramac/Lightweight-Segmentation)
- [DFANet](https://arxiv.org/abs/1904.02216)

[DETAILS](https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/docs/DETAILS.md) for model & backbone.
```
.{SEG_ROOT}
├── core
│   ├── models
│   │   ├── bisenet.py
│   │   ├── danet.py
│   │   ├── deeplabv3.py
│   │   ├── deeplabv3+.py
│   │   ├── denseaspp.py
│   │   ├── dunet.py
│   │   ├── encnet.py
│   │   ├── fcn.py
│   │   ├── pspnet.py
│   │   ├── icnet.py
│   │   ├── enet.py
│   │   ├── ocnet.py
│   │   ├── ccnet.py
│   │   ├── psanet.py
│   │   ├── cgnet.py
│   │   ├── espnet.py
│   │   ├── lednet.py
│   │   ├── dfanet.py
│   │   ├── ......
```

#### Dataset

You can run script to download dataset, such as:

```
cd ./core/data/downloader
python ade20k.py --download-dir ../datasets/ade
```

|                           Dataset                            | training set | validation set | testing set |
| :----------------------------------------------------------: | :----------: | :------------: | :---------: |
| [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) |     1464     |      1449      |      ✘      |
| [VOCAug](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) |    11355     |      2857      |      ✘      |
| [ADK20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/) |    20210     |      2000      |      ✘      |
| [Cityscapes](https://www.cityscapes-dataset.com/downloads/)  |     2975     |      500       |      ✘      |
| [COCO](http://cocodataset.org/#download)           |              |                |             |
| [SBU-shadow](http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip) |     4085     |      638       |      ✘      |
| [LIP(Look into Person)](http://sysu-hcp.net/lip/)       |    30462     |     10000      |    10000    |

```
.{SEG_ROOT}
├── core
│   ├── data
│   │   ├── dataloader
│   │   │   ├── ade.py
│   │   │   ├── cityscapes.py
│   │   │   ├── mscoco.py
│   │   │   ├── pascal_aug.py
│   │   │   ├── pascal_voc.py
│   │   │   ├── sbu_shadow.py
│   │   └── downloader
│   │       ├── ade20k.py
│   │       ├── cityscapes.py
│   │       ├── mscoco.py
│   │       ├── pascal_voc.py
│   │       └── sbu_shadow.py
```

## Result
- **PASCAL VOC 2012**

|Methods|Backbone|TrainSet|EvalSet|crops_size|epochs|JPU|Mean IoU|pixAcc|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN32s|vgg16|train|val|480|60|✘|47.50|85.39|
|FCN16s|vgg16|train|val|480|60|✘|49.16|85.98|
|FCN8s|vgg16|train|val|480|60|✘|48.87|85.02|
|FCN32s|resnet50|train|val|480|50|✘|54.60|88.57|
|PSPNet|resnet50|train|val|480|60|✘|63.44|89.78|
|DeepLabv3|resnet50|train|val|480|60|✘|60.15|88.36|

Note: `lr=1e-4, batch_size=4, epochs=80`.

## Overfitting Test
See [TEST](https://github.com/Tramac/Awesome-semantic-segmentation-pytorch/tree/master/tests) for details.

```
.{SEG_ROOT}
├── tests
│   └── test_model.py
```

## To Do
- [x] add train script
- [ ] remove syncbn
- [ ] train & evaluate
- [x] test distributed training
- [x] fix syncbn ([Why SyncBN?](https://tramac.github.io/2019/04/08/SyncBN/))
- [x] add distributed ([How DIST?](https://tramac.github.io/2019/04/22/%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83-PyTorch/))

## References
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
- [gloun-cv](https://github.com/dmlc/gluon-cv)
- [imagenet](https://github.com/pytorch/examples/tree/master/imagenet)

<!--
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]
-->

[python-image]: https://img.shields.io/badge/Python-2.x|3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.1-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: https://img.shields.io/badge/Apache-2.0-blue.svg
[lic-url]: https://github.com/Tramac/Awesome-semantic-segmentation-pytorch/blob/master/LICENSE
