import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import math
import matplotlib.pyplot as plt

from .detector import build_detector
from .deep_sort import build_tracker
from .utils.draw import draw_boxes
from .utils.parser import get_config
from .utils.log import get_logger
from .utils.io import write_results


def drawcurve1(input):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #
    frame = []
    x = []
    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)

    for m in range(len(input)):
        tem = input[m]
        frame.append(tem[0])
        x.append(tem[1][0][0] + tem[1][0][2] / 2)

    a.plot(frame, x)
    a.set_title('横坐标变化曲线')
    a.set_xlabel('帧序列')
    a.set_ylabel('横坐标')
    # plt.savefig('result1.png')
    # plt.show()


def drawcurve2(input):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #
    frame = []
    y = []
    fig = plt.figure()

    b = fig.add_subplot(1, 1, 1)

    for m in range(len(input)):
        tem = input[m]
        frame.append(tem[0])
        y.append(tem[1][0][1] + tem[1][0][3] / 2)

    b.plot(frame, y)
    b.set_title('纵坐标变化曲线')
    b.set_xlabel('帧序列')
    b.set_ylabel('纵坐标')
    # plt.savefig('result2.png')
    # plt.show()


def drawcurve3(input):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    #
    frame = []
    s = []
    fig = plt.figure()
    c = fig.add_subplot(1, 1, 1)

    for m in range(len(input)):
        tem = input[m]
        frame.append(tem[0])
        s.append(tem[1][0][2] * tem[1][0][3])

    c.plot(frame, s)
    c.set_title('识别框大小变化曲线')
    c.set_xlabel('帧序列')
    c.set_ylabel('识别框大小')
    # plt.savefig('result3.png')
    # plt.show()


class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, str(self.args.proc_id) + ".mp4")
            self.save_results_path = os.path.join(self.args.save_path, str(self.args.proc_id) + ".txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        res = None
        results = []
        mxdifference = []
        mydifference = []
        sizedifference = []
        idx_frame = 0
        while self.vdo.grab():
            idx_frame = idx_frame + 1
            print(idx_frame)
            # if idx_frame % self.args.frame_interval:
            #     continue

            start = time.time()
            return_value, ori_im = self.vdo.retrieve()
            if return_value == True:

                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

                # do detection
                bbox_xywh, cls_conf, cls_ids = self.detector(im)

                # select  class
                bicycle = cls_ids == 1
                car = cls_ids == 2
                motorbike = cls_ids == 3
                bus = cls_ids == 5
                train = cls_ids == 6
                truck = cls_ids == 7
                mask = car + bus + truck + bicycle + motorbike + train

                # person=cls_ids==0
                # mask=person

                bbox_xywh = bbox_xywh[mask]
                # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
                bbox_xywh[:, 3:] *= 1.2
                cls_conf = cls_conf[mask]

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_tlwh = []  # 左上角坐标，横长，竖宽
                    bbox_xyxy = outputs[:, :4]  # 跟踪框左上角和右下角坐标
                    # print(bbox_xyxy)
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                    results.append((idx_frame, bbox_tlwh, identities))

                # 工作状态判断算法，根据识别框中心坐标移动距离以及识别框大小变化共同判断
                if idx_frame == 1 or idx_frame == 2:
                    state = "static"
                    Coordinate = "coordinate:"
                    Size = "size:"
                missframenum = 0
                for i in range(len(results)):
                    Coordinate = "coordinate:"
                    Size = "size:"
                    current = results[i]
                    currentframe = current[0]
                    currentmx = current[1][0][0] + current[1][0][2] / 2
                    currentmy = current[1][0][1] + current[1][0][3] / 2
                    currentsize = current[1][0][2] * current[1][0][3]
                    if currentframe == idx_frame:
                        print(current)
                        zuobiao = round(currentmx, 2), round(currentmy, 2)
                        Coordinate = "coordinate:" + str(zuobiao)
                        Size = "size:" + str(round(currentsize, 2))
                        if currentframe == 3:
                            state = "static"
                            shangyizhenmx = currentmx
                            shangyizhenmy = currentmy
                            shangyizhensize = currentsize
                            print(state)
                        elif currentframe % 30 == 3:
                            dangqianzhenmx = currentmx
                            dangqianzhenmy = currentmy
                            dangqianzhensize = currentsize
                            mxdifference.append(dangqianzhenmx - shangyizhenmx)
                            mydifference.append(dangqianzhenmy - shangyizhenmy)
                            sizedifference.append(dangqianzhensize - shangyizhensize)
                            if math.sqrt((dangqianzhenmx - shangyizhenmx) * (dangqianzhenmx - shangyizhenmx)
                                         + (dangqianzhenmy - shangyizhenmy) * (
                                                 dangqianzhenmy - shangyizhenmy)) >= 4.609:
                                zuobiaoflag = 0.5
                            else:
                                zuobiaoflag = 0
                            if abs(dangqianzhensize - shangyizhensize) >= 3151:
                                sizeflag = 0.5
                            else:
                                sizeflag = 0

                            if (zuobiaoflag + sizeflag) >= 0.5:
                                state = "move"
                            else:
                                state = "static"

                            shangyizhenmx = dangqianzhenmx
                            shangyizhenmy = dangqianzhenmy
                            shangyizhensize = dangqianzhensize
                        else:
                            last = results[i - 1]
                            lastframe = last[0]
                            lastmx = last[1][0][0] + last[1][0][2] / 2
                            lastmy = last[1][0][1] + last[1][0][3] / 2
                            lastsize = last[1][0][2] * last[1][0][3]
                            if currentframe != (lastframe + 1):
                                for j in range(lastframe + 1, currentframe):
                                    if j % 30 == 3:
                                        missframenum = missframenum + 1
                                if missframenum == 1:
                                    dangqianzhenmx = currentmx
                                    dangqianzhenmy = currentmy
                                    dangqianzhensize = currentsize
                                    mxdifference.append(dangqianzhenmx - shangyizhenmx)
                                    mydifference.append(dangqianzhenmy - shangyizhenmy)
                                    sizedifference.append(dangqianzhensize - shangyizhensize)
                                    if math.sqrt((dangqianzhenmx - shangyizhenmx) * (dangqianzhenmx - shangyizhenmx)
                                                 + (dangqianzhenmy - shangyizhenmy) * (
                                                         dangqianzhenmy - shangyizhenmy)) >= 4.609:
                                        zuobiaoflag = 0.5
                                    else:
                                        zuobiaoflag = 0
                                    if abs(dangqianzhensize - shangyizhensize) >= 3151:
                                        sizeflag = 0.5
                                    else:
                                        sizeflag = 0

                                    if (zuobiaoflag + sizeflag) >= 0.5:
                                        state = "move"
                                    else:
                                        state = "static"

                                    shangyizhenmx = dangqianzhenmx
                                    shangyizhenmy = dangqianzhenmy
                                    shangyizhensize = dangqianzhensize
                                if missframenum == 2:
                                    dangqianzhenmx = currentmx
                                    dangqianzhenmy = currentmy
                                    dangqianzhensize = currentsize
                                    mxdifference.append(dangqianzhenmx - lastmx)
                                    mydifference.append(dangqianzhenmy - lastmy)
                                    sizedifference.append(dangqianzhensize - lastsize)
                                    if math.sqrt((dangqianzhenmx - shangyizhenmx) * (dangqianzhenmx - shangyizhenmx)
                                                 + (dangqianzhenmy - shangyizhenmy) * (
                                                         dangqianzhenmy - shangyizhenmy)) >= 4.609:
                                        zuobiaoflag = 0.5
                                    else:
                                        zuobiaoflag = 0
                                    if abs(dangqianzhensize - shangyizhensize) >= 3151:
                                        sizeflag = 0.5
                                    else:
                                        sizeflag = 0

                                    if (zuobiaoflag + sizeflag) >= 0.5:
                                        state = "move"
                                    else:
                                        state = "static"

                                    shangyizhenmx = dangqianzhenmx
                                    shangyizhenmy = dangqianzhenmy
                                    shangyizhensize = dangqianzhensize

                # print(mxdifference)
                # print(mydifference)
                # print(sizedifference)
                # print(state)
                cv2.putText(ori_im, text=Coordinate, org=(20, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8, color=(0, 0, 255), thickness=2)
                cv2.putText(ori_im, text=Size, org=(20, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8, color=(0, 0, 255), thickness=2)
                cv2.putText(ori_im, text=state, org=(20, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8, color=(0, 0, 255), thickness=2)
                # print(results)
                # print(results[0])#(3, [(134, 342, 417, 363)], array([1]))
                # print(results[0][0])#3
                # print(results[0][1])#[(134, 342, 417, 363)]
                # print(results[0][1][0][0])#134
                # print(results[0][2])#[1]
                # print(results[0][2][0])#1
                end = time.time()

                if self.args.display:
                    cv2.imshow("test", ori_im)
                    cv2.waitKey(1)

                if self.args.save_path:
                    self.writer.write(ori_im)

                # save results
                res = write_results(self.save_results_path, results, 'mot')

                # logging
                self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                                 .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))
            else:
                break
            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 画结果曲线
        # drawcurve1(results)
        # drawcurve2(results)
        # drawcurve3(results)
        # for m in range(len(results)):
        #     tem = results[m]
        #     print(tem)
        #     print(tem[0])
        #     print(tem[1])
        #     print(tem[1][0][0])
        return res


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default=os.path.join(os.path.dirname(__file__),
                                                                             'configs/yolov3.yaml'))
    parser.add_argument("--config_deepsort", type=str, default=os.path.join(os.path.dirname(__file__),
                                                                            'configs/deep_sort.yaml'))
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default=os.path.join(os.path.dirname(__file__), "output/"))
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


def excavator_start_recog(vid_path, proc_id):
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)
    args.proc_id = proc_id
    with VideoTracker(cfg, args, video_path=vid_path) as vdo_trk:
        return vdo_trk.run()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
