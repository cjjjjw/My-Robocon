import math
import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import random
import torch
import cv2
import time
from collections import deque
import multiprocessing
from Map_Reflect_utils import if_model, Map_Reflect
import pandas as pd
import gc

# 定义棋子
bluecode = 1.0
redcode = -1.0
backcode = 0.0

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms,
                                 copy_paste,
                                 letterbox, mixup, random_perspective)
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements,
                           colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer,
                           xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode, time_sync


def lib_init():
    global DetectMultiBackend
    global Annotator, colors, save_one_box
    from models.common import DetectMultiBackend
    from utils.plots import Annotator, colors, save_one_box


# 数组的裁切
def delete(arr):
    SHAPE = arr.shape
    DELETE = 0
    arr[:, DELETE] = arr[:, SHAPE[1] - 1]
    arr2 = arr[:, :SHAPE[1] - 1].T
    arr2 = np.squeeze(arr2).tolist()
    return arr2


# 将整数四舍五入，方便做聚类
def round_int(nums: int, n: int):
    """
    :param nums: 需要四舍五入的数
    :param n: 四舍五入从后到前的第几位
    :return: 四舍五入后的数
    """
    if nums < pow(10, n):
        return False
    end_pos = int((nums % pow(10, n) - nums % pow(10, n - 1)) / pow(10, n - 1))
    if end_pos >= 5:
        nums += pow(10, n) - nums % pow(10, n)
    else:
        nums -= nums % pow(10, n)
    return nums


def FUCKING(queue, share_var, shape_x):
    # 先进行按列排序
    color_code = bluecode
    env = Map_Reflect.State()
    env.map_init()  # 地图启动
    model = if_model.Judgment
    share_var['Reflect_img'] = env.map
    while True:
        # 打开管道接收数据
        mediator_dict = queue.get()
        if mediator_dict["ball_deque_coord"]:
            data_coord = mediator_dict["ball_deque_coord"]
            x_coordinate = sorted(mediator_dict['col'])
            data_coord = np.array(data_coord)
            cluster = deque(
                [deque([[0, 0], [0, 0], [0, 0]]), deque([[0, 0], [0, 0], [0, 0]]), deque([[0, 0], [0, 0], [0, 0]]),
                 deque([[0, 0], [0, 0], [0, 0]]), deque([[0, 0], [0, 0], [0, 0]])])
            Final_cluster = deque()
            if len(x_coordinate) == 5:
                env.reset()
                # 排序
                data_coord = data_coord[np.argsort(data_coord[:, 0])].tolist()
                # 为了提升速度，可以直接将下面的转换成数组变量
                # for i in range(5):
                #     cluster.append(deque())
                #     for x in range(3):
                #         cluster[i].append([0, 0])
                for i in range(len(data_coord)):
                    for x in range(len(x_coordinate)):
                        # print(data_coord,mediator_dict["height"])
                        if abs(data_coord[i][0] - x_coordinate[x]) < mediator_dict["width"] / 2 and abs(
                                data_coord[i][1]) > \
                                mediator_dict["height"] - (mediator_dict["width"] / 2) / 1.2:
                            cluster[x].append(data_coord[i][1:])
                            # print(data_coord[i][0], x)
                            del cluster[x][0]
                cluster = np.array(cluster)
                x_coordinate = abs(np.array(sorted(mediator_dict["col"])) - shape_x / 2)
                # print(x_coordinate, "x_coord")
                for i in range(len(cluster)):
                    # 下列是映射和·做决策的过程
                    cluster[i] = cluster[i][np.argsort(cluster[i][:, 0])]
                    Final_cluster.append(delete(cluster[i]))
                Final_cluster = np.array(Final_cluster).T
                env.reflect(Final_cluster)
                try:
                    action, min_x_best = model(x_coordinate, Final_cluster, color_code).main_decision()
                    next_state, _, _ = env.step(action, 2)
                    action2, min_x_better = model(x_coordinate, next_state, color_code).main_decision()
                    next_state, _, _ = env.step(action2, 3)
                    # P1(x,y),P2(x,y),
                    # 复原最佳的框，对速度没什么影响，但是不在主进程获取ROI
                    Best_ROI = (int((min_x_best + shape_x / 2) - (mediator_dict["width"] / 2)),
                                int(mediator_dict["height"]),
                                int(mediator_dict["width"]),
                                int(mediator_dict["low"] - mediator_dict["height"]))
                    # print(action2, 'better action')
                    share_var['Best'] = action
                    share_var['Better'] = action2
                    share_var['Best_ROI'] = Best_ROI
                except Exception:
                    print("No other situation")
        share_var['Reflect_img'] = env.map


def KFC_Thursday(stack, share_var):
    global tracker
    track_target_last = None
    update_counter = 0
    # 留下一个Track_flag,与下位机做交流
    Track_flag = True
    while True:
        Best_ROI_now = share_var["Best_ROI"]
        track_target_now = share_var["Best_target"]
        track_flag = share_var["Track_flag"]
        if len(stack) != 0:
            print(update_counter)
            Frame = stack.pop()
            if track_flag:
                update_counter += 1
                if track_target_now != track_target_last and update_counter < 50 and track_target_now:
                    track_target_last = track_target_now
                    tracker = cv2.TrackerKCF.create()
                    tracker.init(Frame, Best_ROI_now)
                    update_counter = 0
                    # 避免同时更新和初始化
                elif update_counter > 50 and track_target_last:
                    # 当锁定目标后，即只会更新KCF的瞄框，而不会更新KCF的状态
                    update_counter = 50
                    status, coord = tracker.update(Frame)
                    if status:
                        print("successfully locking")
                        mid_pos = (int(coord[0] + coord[2] / 2), int(coord[1] + coord[3] / 2))
                        p1 = (int(coord[0]), int(coord[1]))
                        p2 = (int(coord[0] + coord[2]), int(coord[1] + coord[3]))
                        share_var["Track_data"] = [mid_pos, p1, p2]
                elif update_counter > 100:
                    update_counter = 0
                    print("long time no see")
            else:
                print("No target")
                track_target_last = None
                update_counter = 0
                share_var["Track_data"] = [(0, 0), None, None]


def cap_capture(stack, stack2, source, top, top2) -> None:
    """
    :param top2: 缓冲栈容量
    :param stack2: Manager.list对象,作为第二通道的视频流
    :param source: 摄像头参数
    :param stack: Manager.list对象,作为第一通道的视频流
    :param top: 缓冲栈容量
    :return: None
    """
    global cap
    if len(source) < 2:
        for i in range(10):
            try:
                cap = cv2.VideoCapture(int(source))
            except Exception:
                pass
            if cap.isOpened():
                break
    else:
        cap = cv2.VideoCapture(source)
    while True:
        ret, Frame = cap.read()
        if ret:
            stack.append(Frame)
            stack2.append(Frame)
            if len(stack) >= top:
                del stack[:]
            if len(stack2) >= top2:
                del stack2[:]


class Track(object):
    # 继承plot的类,传入一个共享变量在并行处理的时候可以读取到BucketTime的各种信息
    def __init__(self):
        # todo 设立一个中介字典，存储并转换球，之后用一个聚类算法，将x作为聚类分为5簇，y则直接排序，并且当col到一定量的时候才会更新这个字典
        self.share_data = dict
        self.mediator_dict = {"col": deque(), "ball_deque_coord": deque(), "width": int(), "height": int(),
                              "low": int()}

    # 修改图像的对比度,coefficent>0, <1降低对比度,>1提升对比度 建议0-2
    def change_contrast(self, img, coefficent):
        imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m = cv2.mean(img)[0]
        graynew = m + coefficent * (imggray - m)
        img1 = np.zeros(img.shape, np.float32)
        k = np.divide(graynew, imggray, out=np.zeros_like(graynew), where=imggray != 0)
        img1[:, :, 0] = img[:, :, 0] * k
        img1[:, :, 1] = img[:, :, 1] * k
        img1[:, :, 2] = img[:, :, 2] * k
        img1[img1 > 255] = 255
        img1[img1 < 0] = 0
        return img1.astype(np.uint8)

    # 修改图像的亮度，brightness取值0～2 <1表示变暗 >1表示变亮
    def change_brightness(self, img, brightness):
        [averB, averG, averR] = np.array(cv2.mean(img))[:-1] / 3
        k = np.ones(img.shape)
        k[:, :, 0] *= averB
        k[:, :, 1] *= averG
        k[:, :, 2] *= averR
        img = img + (brightness - 1) * k
        img[img > 255] = 255
        img[img < 0] = 0
        return img.astype(np.uint8)

    def run_normal(self, weights=ROOT / 'yolov5s.pt',  # model path or triton URL
                   source='data/images',  # file/dir/URL/glob/screen/0(webcam)
                   data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
                   imgsz=(640, 640),  # inference size (height, width)
                   conf_thres=0.25,  # confidence threshold
                   iou_thres=0.45,  # NMS IOU threshold
                   max_det=1000,  # maximum detections per image
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   classes=None,  # filter by class: --class 0, or --class 0 2 3
                   agnostic_nms=False,  # class-agnostic NMS
                   augment=False,  # augmented inference
                   line_thickness=3,
                   half=False,  # 半精度模型
                   dnn=True,  # opencv dnn 加速
                   ):
        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # fp16半精度模型，dnn 加速推理
        stride, names = model.stride, model.names
        imgsz = check_img_size(imgsz, s=stride)  # check image size,确定推理照片尺寸
        imgs = [None]  # 齐次化做准备
        while True:
            if len(Cap_data) != 0:
                self.mediator_dict["ball_deque_coord"], self.mediator_dict["col"], self.mediator_dict[
                    "width"], self.mediator_dict["low"] = deque(), deque(), 0, 0
                imgs[0] = Cap_data.pop()
                # copy to become another image
                im0s = imgs.copy()
                img = [letterbox(x, new_shape=imgsz)[0] for x in im0s]
                img = np.stack(img, 0)  # 齐次化
                img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to 3x416x416, uint8 to float32
                img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)
                img /= 255.0  # 0 - 255 to 0.0 - 1.0 归一化操作
                im = torch.from_numpy(img).to(model.device)  # transform stype
                # start predict
                pred = model(im, augment=False, visualize=False)  # 不增强任何数据和保存，提高推理速度
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                           max_det=max_det)  # 非最大化抑制，主要为画框
                for i, det in enumerate(pred):  # 枚举每一次预测结果
                    s, im0 = '%g: ' % i, im0s[i].copy()  # s为预测准确率
                    s += '%gx%g ' % img.shape[2:]  # print string
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):  # 判读是否有物体识别到
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # 坐标映射
                        # Print resultssave
                        # 在这个循环里，会一次性将所有的目标物展现
                        for *xyxy, conf, cls in reversed(det):  # 识别物的类别像素坐标，置信度，物体类别号
                            mid_pos = [int((int(xyxy[0]) + int(xyxy[2])) / 2),
                                       int((int(xyxy[1]) + int(xyxy[3])) / 2)]
                            cv2.circle(im0, mid_pos, 1, colors(int(cls)), 3)  # 圆心
                            cv2.putText(im0, f"{mid_pos}", mid_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            if names[int(cls)] == "r":
                                self.mediator_dict["ball_deque_coord"].append([*mid_pos, int(redcode)])
                            elif names[int(cls)] == "b":
                                self.mediator_dict["ball_deque_coord"].append([*mid_pos, int(bluecode)])
                            elif names[int(cls)] == "c":
                                self.mediator_dict["width"] = int((xyxy[2] - xyxy[0]))
                                self.mediator_dict["height"] = int(xyxy[1])
                                self.mediator_dict["low"] = int(xyxy[3])
                                self.mediator_dict["col"].append(mid_pos[0])
                            label = '%s %.2f' % (names[int(cls)], conf)
                            annotator.box_label(xyxy, label, color=colors(int(cls), True))
                        queue.put(self.mediator_dict)
                        KCF_data["Track_flag"] = True
                    else:
                        KCF_data["Track_flag"] = False
                    im0 = annotator.result()
                    self.share_data["Reflect_data"] = Reflect_data
                    KCF_data["Best_ROI"] = Reflect_data["Best_ROI"]
                    KCF_data["Best_target"] = Reflect_data["Best"]
                    # 以下是向外输送
                    if KCF_data["Track_data"]:
                        self.share_data["Best_x"] = KCF_data["Track_data"][0][0]
                        cv2.rectangle(im0, KCF_data["Track_data"][1], KCF_data["Track_data"][2], (255, 0, 0), 2, 1)
                    self.share_data["im0"] = im0

    # This function is used to choose argument,which is called init function
    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/best.pt',
                            help='model path or triton URL')
        parser.add_argument('--source', type=str, default="1",
                            help='')
        parser.add_argument('--data', type=str, default=ROOT / 'data/ball.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.51, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.55, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=100, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--half', default=False, action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', default=True, action='store_true', help='use OpenCV DNN for ONNX inference')
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt

    # start function
    def main(self, shareVar):
        global queue, Reflect_data, Cap_data, KCF_data
        queue = multiprocessing.Queue()
        Reflect_data = multiprocessing.Manager().dict()
        KCF_data = multiprocessing.Manager().dict()
        # 创建两条视频流
        Cap_data = multiprocessing.Manager().list()
        Cap_data2 = multiprocessing.Manager().list()
        # 初始化赋值["Best_ROI"],Reflect_data["Best_ROI"]避免先启动读不到数据
        # share_var['Best_track']
        KCF_data["Best_ROI"], Reflect_data["Best_ROI"], KCF_data["Track_data"], Reflect_data["Best"], KCF_data[
            "Best_target"], KCF_data["Track_flag"] = (), (), [], None, None, False
        # 为了将数据送出，构建一个共享字典
        self.share_data = shareVar
        # 使其为全局变量，不然无法添加
        opt = self.parse_opt()
        check_requirements(exclude=('tensorboard', 'thop'))
        lib_init()
        # 启动进程
        p1 = multiprocessing.Process(target=cap_capture, args=(Cap_data, Cap_data2, vars(opt)["source"], 15, 100),
                                     daemon=True)
        p2 = multiprocessing.Process(target=FUCKING, args=(queue, Reflect_data, 640), daemon=True)
        p3 = multiprocessing.Process(target=KFC_Thursday, args=(Cap_data2, KCF_data), daemon=True)
        p1.start()
        p2.start()
        p3.start()
        # 便于关闭所有进程
        self.run_normal(**vars(opt))


if __name__ == "__main__":
    shareVar = multiprocessing.Manager().dict()
    t_1 = multiprocessing.Process(target=Track().main, args=(shareVar,))
    t_1.start()
    while True:
        try:
            cv2.imshow("1231", shareVar['im0'])
            cv2.imshow('321', shareVar["Reflect_data"]["Reflect_img"])
            # print('最佳选择', shareVar["Reflect_data"]["Best"])
            # print('次要选择', shareVar["Reflect_data"]["Better"])
            # print('最佳选择_x', shareVar["Best_x"])
            cv2.waitKey(1)
        except Exception:
            pass
