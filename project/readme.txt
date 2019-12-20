环境配置：
·将VOCdevkit文件夹和checkpoints文件夹（单独拷给了助教）放到主文件夹下
·请安装requirements.txt中的库，使用pytorch 0.3.0版本

--------------------------------------------------------------------------

使用方法：
·使用某一个model：
在utils/config.py中，设置load_path为checkpoint文件地址
在model文件夹下，把checkpoint对应的fast rcnn的py文件改名为fast_rcnn.py
checkpoints的格式是：
模型名称_生成时间_对应epoch_mAP

·生成一张图的object detection结果：
使用某一个model，读取对应checkpoints
将图片命名为1.jpg，放在主文件下
python test.py
生成ans.jpg

·训练：
python train.py train
会训练18个epoch，其中第10个epoch开始进行evaluation，输出mAP并储存checkpoints

·训练的过程中plot loss、pred_img和gt_img：
在训练前请：
python -m visdom.server
打开浏览器，访问http://localhost:{config中设置的端口号}
plot出的loss是当前epoch到当前时间点的loss的平均

--------------------------------------------------------------------------

project的逻辑和结构：
·如图procedure.jpg所示
·使用pytorch实现，使用PASCAL VOC数据集，dataset.py负责提供数据，trainer.py负责在训练时按顺序forward整个Faster RCNN的各个部分，并且求各部分的loss，对应faster_rcnn.py里的forward无需计算loss

·Faster RCNN主要由三个部分组成：
VGG16extractor（feature_extractor.py），RPN（region_proposal_network.py），ROIHead（fast_rcnn_{model编号}.py）

·用VGG16（torchvision pretrained）提取feature map

·RPN得到feature map后，在feature map每个点放置九个不同比例（0.5, 1, 2）、不同大小（1 : 2 : 4）的Anchor，通过位置回归、对象分类（对象/背景二分类）和NMS（model/util/nms文件夹）给出rois（regions of interest）

·ROIHead（Fast RCNN）对每个roi通过Roi Pooling（roi_module.py）给出512 * 7 * 7的feature

·Fast RCNN生成feature后的处理过程和model类型相关：
原始Faster RCNN和用于对照的加一层FC的Faster RCNN不走relation module
其余model分别走1层/2层relation layer、含/不含位置信息、1个/4个/16个relation block的relation module
编号和model的对应关系如下：
raw		原始Faster RCNN
1		有位置 1个relation 1层
2		无位置 1个relation 1层
3		无位置 1个relation 2层
4		无位置 4个relation 1层
5		无位置 4个relation 2层
6		有位置 1个relation 2层
7		有位置 4个relation 1层
8		有位置 4个relation 2层
9		raw +1 层
10		有位置 16个relation 2层
11		无位置 16个relation 2层
12		有位置 16个relation 2层 加report中EQ(6)的位置embedding

·为了提升速度，训练时采用联合训练（而非Faster RCNN论文中提到的分步训练）：
total_loss = rpn_loc_loss + rpn_cls_loss + fast_rcnn_loc_loss + fast_rcnn_cls_loss
