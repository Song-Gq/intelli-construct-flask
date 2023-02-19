import xml.etree.ElementTree as ET
from os import getcwd
from utils.utils import get_classes

annotation_mode     = 0

classes_path        = 'VOCdevkit/VOC2022/class.txt'

VOCdevkit_path  = 'VOCdevkit'
sets            = [('2022', 'train'), ('2022', 'val')]
classes =get_classes(classes_path)

def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id),'rb')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text

        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()


# ##训练2008的权重
# import xml.etree.ElementTree as ET
# from os import getcwd
# from utils.utils import get_classes
#
# annotation_mode     = 0
#
# classes_path        = 'VOCdevkit/VOC2008/class.txt'
# trainval_percent    = 0.9
# train_percent       = 0.9
# '''
# 指向VOC数据集所在的文件夹
# 默认指向根目录下的VOC数据集
# '''
# VOCdevkit_path  = 'VOCdevkit'
# sets            = [('2008', 'train'), ('2008', 'val')]
# classes =get_classes(classes_path)
#
# def convert_annotation(year, image_id, list_file):
#     in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id),'rb')
#     tree=ET.parse(in_file)
#     root = tree.getroot()
#
#     for obj in root.iter('object'):
#         difficult = 0
#         if obj.find('difficult')!=None:
#             difficult = obj.find('difficult').text
#
#         cls = obj.find('name').text
#         if cls not in classes or int(difficult)==1:
#             continue
#         cls_id = classes.index(cls)
#         xmlbox = obj.find('bndbox')
#         b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
#         list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
#
# wd = getcwd()
#
# for year, image_set in sets:
#     image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
#     list_file = open('%s_%s.txt'%(year, image_set), 'w')
#     for image_id in image_ids:
#         list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
#         convert_annotation(year, image_id, list_file)
#         list_file.write('\n')
#     list_file.close()



# ##训练2007得权重
# import xml.etree.ElementTree as ET
# from os import getcwd
# from utils.utils import get_classes
#
# '''
# annotation_mode用于指定该文件运行时计算的内容
# annotation_mode为0代表整个标签处理过程，包括获得VOCdevkit/VOC2007/ImageSets里面的txt以及训练用的2007_train.txt、2007_val.txt
# annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets里面的txt
# annotation_mode为2代表获得训练用的2007_train.txt、2007_val.txt
# '''
# annotation_mode     = 0
#
# classes_path        = 'model_data/predefined_classes.txt'
# '''
# trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
# train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
# 仅在annotation_mode为0和1的时候有效
# '''
# trainval_percent    = 0.9
# train_percent       = 0.9
# '''
# 指向VOC数据集所在的文件夹
# 默认指向根目录下的VOC数据集
# '''
# VOCdevkit_path  = 'VOCdevkit'
# sets            = [('2007', 'train'), ('2007', 'val')]
# classes =get_classes(classes_path)
#
# # sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
# # classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#
#
# def convert_annotation(year, image_id, list_file):
#     in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
#     tree=ET.parse(in_file)
#     root = tree.getroot()
#
#     for obj in root.iter('object'):
#         difficult = 0
#         if obj.find('difficult')!=None:
#             difficult = obj.find('difficult').text
#
#         cls = obj.find('name').text
#         if cls not in classes or int(difficult)==1:
#             continue
#         cls_id = classes.index(cls)
#         xmlbox = obj.find('bndbox')
#         b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
#         list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
#
# wd = getcwd()
#
# for year, image_set in sets:
#     image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
#     list_file = open('%s_%s.txt'%(year, image_set), 'w')
#     for image_id in image_ids:
#         list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
#         convert_annotation(year, image_id, list_file)
#         list_file.write('\n')
#     list_file.close()


