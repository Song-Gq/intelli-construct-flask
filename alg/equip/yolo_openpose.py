import os

import cv2
import numpy as np
import colorsys
import tensorflow as tf

from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from PIL import Image, ImageFont, ImageDraw
from alg.equip.nets.yolo4 import yolo_body,yolo_eval
from alg.equip.utils.utils import letterbox_image

import pandas as pd


def label_openpose(df):
    H_pose = []
    V_pose = []
    # 图片中的第i个人
    for i in range(len(df)):
        A = df[i]['box_pose_pos']
        # 四点法hat
        a_17 = A[17]
        a_18 = A[18]
        a_1 = A[1]
        a_8 = A[8]
        a_2 = A[2]
        a_5 = A[5]
        # 当点未检测到时
        if a_17 != [0, 0] and a_18 != [0, 0]:
            M_x = (a_17[0] + a_18[0]) / 2
            M_y = (a_17[1] + a_18[1]) / 2
            if a_2 != [0, 0] and a_5 != [0, 0]:
                r1 = pow(pow(M_x - a_5[0], 2) + pow(M_y - a_5[1], 2), 0.5)
                r2 = pow(pow(M_x - a_2[0], 2) + pow(M_y - a_2[1], 2), 0.5)
                r = max(r1, r2)
            elif a_2 == [0, 0]:
                r = pow(pow(M_x - a_5[0], 2) + pow(M_y - a_5[1], 2), 0.5)
            else:
                r = pow(pow(M_x - a_2[0], 2) + pow(M_y - a_2[1], 2), 0.5)

        elif a_17 == [0, 0]:
            M_x, M_y = a_18
            if a_2 != [0, 0] and a_5 != [0, 0]:
                r1 = pow(pow(M_x - a_5[0], 2) + pow(M_y - a_5[1], 2), 0.5)
                r2 = pow(pow(M_x - a_2[0], 2) + pow(M_y - a_2[1], 2), 0.5)
                r = max(r1, r2)
            elif a_2 == [0, 0]:
                r = pow(pow(M_x - a_5[0], 2) + pow(M_y - a_5[1], 2), 0.5)
            else:
                pow(pow(M_x - a_2[0], 2) + pow(M_y - a_2[1], 2), 0.5)

        else:
            M_x, M_y = a_17
            if a_2 != [0, 0] and a_5 != [0, 0]:
                r1 = pow(pow(M_x - a_5[0], 2) + pow(M_y - a_5[1], 2), 0.5)
                r2 = pow(pow(M_x - a_2[0], 2) + pow(M_y - a_2[1], 2), 0.5)
                r = max(r1, r2)
            elif a_2 == [0, 0]:
                r = pow(pow(M_x - a_5[0], 2) + pow(M_y - a_5[1], 2), 0.5)
            else:
                r = pow(pow(M_x - a_2[0], 2) + pow(M_y - a_2[1], 2), 0.5)

        box_hat = [int(M_y - r), int(M_x - r), int(M_y + r), int(M_x + r)]
        H_pose.append(box_hat)

        # 两点法vest
        if a_8 != [0, 0]:
            M_x = (A[1][0] + A[8][0]) / 2
            M_y = (A[1][1] + A[8][1]) / 2
            r = pow(pow(M_x - a_1[0], 2) + pow(M_y - a_1[1], 2), 0.5)
            box_vest = [int(M_y - r), int(M_x - r), int(M_y + r), int(M_x + r)]

        # 三点法vest
        else:
            M_x = a_1[0]
            M_y = a_1[1]
            r1 = pow(pow(M_x - a_5[0], 2) + pow(M_y - a_5[1], 2), 0.5)
            r2 = pow(pow(M_x - a_2[0], 2) + pow(M_y - a_2[1], 2), 0.5)
            r = max(r1, r2)
            box_vest = [int(M_y - r), int(M_x - r), int(M_y + r), int(M_x + r)]
        V_pose.append(box_vest)
    H = np.maximum(H_pose, 0)
    V = np.maximum(V_pose, 0)
    return H, V

def pose_iou(gt_box, b_box):
    width0 = gt_box[2] - gt_box[0]  # 矩形的宽度
    height0 = gt_box[3] - gt_box[1]  # 矩形的高度
    width1 = b_box[2] - b_box[0]
    height1 = b_box[3] - b_box[1]
    max_x = max(gt_box[2], b_box[2])
    min_x = min(gt_box[0], b_box[0])
    width = max(width0 + width1 - (max_x - min_x),0)
    max_y = max(gt_box[3], b_box[3])
    min_y = min(gt_box[1], b_box[1])
    height = max(height0 + height1 - (max_y - min_y),0)

    interArea = max(width * height, 0)
    boxAArea = width0 * height0
    boxBArea = width1 * height1
    # iou = interArea / (boxAArea + boxBArea - interArea)
    iou = interArea / min(boxAArea, boxBArea)
    return iou

def label_yolo(W, image):
    top, left, bottom, right = W
    top = top - 5
    left = left - 5
    bottom = bottom + 5
    right = right + 5
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
    A = [top, left, bottom, right]
    return A


class YOLO(object):
    _defaults = {
        # VOC2022
        "model_path": 'alg/equip/logs/ep050-loss75.476-val_loss82.279.h5',
        "anchors_path": 'alg/equip/model_data/yolo_anchors.txt',
        "classes_path": 'alg/equip/model_data/predefined_classes.txt',

        "score": 0.5,
        "iou": 0.3,
        "eager": False,
        "max_boxes": 100,
        "model_image_size": (416, 416)
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        if not self.eager:
            tf.compat.v1.disable_eager_execution()
            self.sess = K.get_session()
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 计算anchor数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # 载入模型，如果原来的模型里已经包括了模型结构则直接载入。
        # 否则先构建模型再载入
        self.yolo_model = yolo_body(Input(shape=(None, None, 3)), num_anchors // 3, num_classes)
        self.yolo_model.load_weights(self.model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)

        if self.eager:
            self.input_image_shape = Input([2, ], batch_size=1)
            inputs = [*self.yolo_model.output, self.input_image_shape]
            outputs = Lambda(yolo_eval, output_shape=(1,), name='yolo_eval',
                             arguments={'anchors': self.anchors, 'num_classes': len(self.class_names),
                                        'image_shape': self.model_image_size,
                                        'score_threshold': self.score, 'eager': True, 'max_boxes': self.max_boxes})(
                inputs)
            self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)
        else:
            self.input_image_shape = K.placeholder(shape=(2,))

            self.boxes, self.scores, self.classes = yolo_eval(self.yolo_model.output, self.anchors,
                                                              num_classes, self.input_image_shape,
                                                              max_boxes=self.max_boxes,
                                                              score_threshold=self.score, iou_threshold=self.iou)

    def outcomes(self, image, df):
        # start = timer()

        # 调整图片使其符合输入要求
        new_image_size = (self.model_image_size[1], self.model_image_size[0])
        boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        if self.eager:
            # 预测结果
            input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
            out_boxes, out_scores, out_classes = self.yolo_model.predict([image_data, input_image_shape])
        else:
            # 预测结果
            out_boxes, out_scores, out_classes = self.sess.run(
                [self.boxes, self.scores, self.classes],
                feed_dict={
                    self.yolo_model.input: image_data,
                    self.input_image_shape: [image.size[1], image.size[0]],
                    K.learning_phase(): 0
                })

        font = ImageFont.truetype(font='alg/equip/font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        H = []
        V = []
        H_pose, V_pose = label_openpose(df)
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            box1 = label_yolo(box, image)
            score = out_scores[i]
            if 1 <= c <= 5:
                H.append([box1, predicted_class])
            if c == 6:
                V.append(box1)

        # 画worker框
        for k in range(len(H_pose)):
            IOU_H = []
            IOU_V = []
            #hat决策输出
            for j in range(len(H)):
                IOU_H.append(pose_iou(H_pose[k], H[j][0]))
            if not IOU_H:
                IOU_H.append(0)
            I_H = max(IOU_H)
            if I_H >= 0.5:
                I_anchor = np.argmax(IOU_H, axis=-1)
                predicted_class = H[I_anchor][1]
                # c = self.class_names.index(H[I_anchor][1])
                if predicted_class=='yellow' or predicted_class=='orange':
                    label='普通工人'
                    c = 2
                elif predicted_class == 'red':
                    label = '管理人员'
                    c = 3
                elif predicted_class == 'blue':
                    label = '技术人员'
                    c = 4
                else :
                    label = '监理'
                    c = 5
            else:
                label = 'no_hat'
                c = 1

            for j in range(len(V)):
                IOU_V.append(pose_iou(V_pose[k], V[j]))
            if not IOU_V:
                IOU_V.append(0)
            I_V = max(IOU_V)
            if I_V >= 0.5:
                if label=='no_hat':
                    label1='不合规：未戴安全帽'
                    c1=1
                else:
                    label1=label
                    c1=c
            else:
                if label == 'no_hat':
                    label1 = '不合规：未戴安全帽,未穿工作服'
                    c1 = 1
                elif label == '普通工人':
                    label1 = '不合规：未穿工作服'
                    c1 = 1
                else:
                    label1 = label
                    c1 = c

            # worker框
            top, left, bottom, right = H_pose[k]

            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label1, font)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for r in range(thickness):
                draw.rectangle(
                    [left + r, top + r, right - r, bottom - r],
                    outline=self.colors[c1])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c1])
            draw.text(text_origin, str(label1), fill=(0, 0, 0), font=font)
            del draw

        # for k1 in range(len(H)):
        #     top, left, bottom, right = H[k1][0]
        #     label1 = H[k1][1]
        #     draw = ImageDraw.Draw(image)
        #     label_size = draw.textsize(label1, font)
        #
        #     if top - label_size[1] >= 0:
        #         text_origin = np.array([left, top - label_size[1]])
        #     else:
        #         text_origin = np.array([left, top + 1])
        #
        #     for r in range(thickness):
        #         draw.rectangle(
        #             [left + r, top + r, right - r, bottom - r],
        #             outline=self.colors[0])
        #     draw.rectangle(
        #         [tuple(text_origin), tuple(text_origin + label_size)],
        #         fill=self.colors[0])
        #     draw.text(text_origin, str(label1), fill=(0, 0, 0), font=font)
        #     del draw

        return image

    def close_session(self):
        self.sess.close()


def equip_start_recog(img_id, img_path, proc_id):
    try:
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        yolo = YOLO()
        # 输出路径
        # outdir = "alg/equip/output/"

        # results.json为openpose读出的json文件
        df = pd.read_json('alg/equip/results.json')

        # 图片输入，测试图片路径：img/test/*.jpg
        # image = "000193.jpg"
        n = df[img_id][0]
        df1 = df[img_id][1:n + 1]
        # img0 = cv2.imread(img_path)
        # img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        # path = 'img/test/' + image
        img0 = Image.open(img_path)
        res_img = yolo.outcomes(img0, df1)
        res_img.save("alg/equip/output/" + str(proc_id) + ".jpg")
        return 0
    except Exception as e:
        print("equip_start_recog(): {}".format(e))
        return -1
