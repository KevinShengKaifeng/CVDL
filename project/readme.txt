�������ã�
����VOCdevkit�ļ��к�checkpoints�ļ��У��������������̣��ŵ����ļ�����
���밲װrequirements.txt�еĿ⣬ʹ��pytorch 0.3.0�汾

--------------------------------------------------------------------------

ʹ�÷�����
��ʹ��ĳһ��model��
��utils/config.py�У�����load_pathΪcheckpoint�ļ���ַ
��model�ļ����£���checkpoint��Ӧ��fast rcnn��py�ļ�����Ϊfast_rcnn.py
checkpoints�ĸ�ʽ�ǣ�
ģ������_����ʱ��_��Ӧepoch_mAP

������һ��ͼ��object detection�����
ʹ��ĳһ��model����ȡ��Ӧcheckpoints
��ͼƬ����Ϊ1.jpg���������ļ���
python test.py
����ans.jpg

��ѵ����
python train.py train
��ѵ��18��epoch�����е�10��epoch��ʼ����evaluation�����mAP������checkpoints

��ѵ���Ĺ�����plot loss��pred_img��gt_img��
��ѵ��ǰ�룺
python -m visdom.server
�������������http://localhost:{config�����õĶ˿ں�}
plot����loss�ǵ�ǰepoch����ǰʱ����loss��ƽ��

--------------------------------------------------------------------------

project���߼��ͽṹ��
����ͼprocedure.jpg��ʾ
��ʹ��pytorchʵ�֣�ʹ��PASCAL VOC���ݼ���dataset.py�����ṩ���ݣ�trainer.py������ѵ��ʱ��˳��forward����Faster RCNN�ĸ������֣�����������ֵ�loss����Ӧfaster_rcnn.py���forward�������loss

��Faster RCNN��Ҫ������������ɣ�
VGG16extractor��feature_extractor.py����RPN��region_proposal_network.py����ROIHead��fast_rcnn_{model���}.py��

����VGG16��torchvision pretrained����ȡfeature map

��RPN�õ�feature map����feature mapÿ������þŸ���ͬ������0.5, 1, 2������ͬ��С��1 : 2 : 4����Anchor��ͨ��λ�ûع顢������ࣨ����/���������ࣩ��NMS��model/util/nms�ļ��У�����rois��regions of interest��

��ROIHead��Fast RCNN����ÿ��roiͨ��Roi Pooling��roi_module.py������512 * 7 * 7��feature

��Fast RCNN����feature��Ĵ�����̺�model������أ�
ԭʼFaster RCNN�����ڶ��յļ�һ��FC��Faster RCNN����relation module
����model�ֱ���1��/2��relation layer����/����λ����Ϣ��1��/4��/16��relation block��relation module
��ź�model�Ķ�Ӧ��ϵ���£�
raw		ԭʼFaster RCNN
1		��λ�� 1��relation 1��
2		��λ�� 1��relation 1��
3		��λ�� 1��relation 2��
4		��λ�� 4��relation 1��
5		��λ�� 4��relation 2��
6		��λ�� 1��relation 2��
7		��λ�� 4��relation 1��
8		��λ�� 4��relation 2��
9		raw +1 ��
10		��λ�� 16��relation 2��
11		��λ�� 16��relation 2��
12		��λ�� 16��relation 2�� ��report��EQ(6)��λ��embedding

��Ϊ�������ٶȣ�ѵ��ʱ��������ѵ��������Faster RCNN�������ᵽ�ķֲ�ѵ������
total_loss = rpn_loc_loss + rpn_cls_loss + fast_rcnn_loc_loss + fast_rcnn_cls_loss
