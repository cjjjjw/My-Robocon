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
from Map_Reflect_utils import if_model, Map_Reflect, Realsense
import pandas as pd
from numba import jit
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("Depth and Reflect")

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
    if SHAPE[0] < 2:
        return arr2
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


@jit(nopython=True)
def accle_for_Y(detections, Y_max=600, Y_min=250):
    detections_ = []
    for i in range(len(detections)):
        Y = detections[i][1]  # 获取Y
        # 第一次进行过滤，过滤掉大部分没用信息，但是设定阈值不能太高
        if Y_min <= Y <= Y_max:
            detections_.append(detections[i])
    return detections_


@jit(nopython=True)
def accle_for_depth(detections, depth_max=0.5, depth_min=0):
    detections_ = []
    for i in range(len(detections)):
        depth = detections[i][4]  # 获取depth
        # 第一次进行过滤，过滤掉大部分没用信息，但是设定阈值不能太高
        if depth_min <= depth <= depth_max:
            detections_.append(detections[i])
    return detections_


@jit(nopython=True)
def accle_for_X(detections, X_max=350, X_min=250):
    detections_ = []
    for i in range(len(detections)):
        X = detections[i][0]  # 获取depth
        # 第一次进行过滤，过滤掉大部分没用信息，但是设定阈值不能太高
        if X_min <= X <= X_max:
            detections_.append(detections[i])
    return detections_


def total_get(detections):
    # detections = np.array(accle_for_Y(detections))
    detections = np.array(accle_for_depth(detections))
    # detections = np.array(accle_for_X(detections))
    return np.array(detections)


def FUCKING(queue, share_var):
    # 先进行按列排序
    color_code = bluecode
    env = Map_Reflect.State()
    env.map_init(False)  # 地图启动
    model = if_model.Judgment
    share_var['Reflect_img'] = env.map
    x_coordinate = np.array([5, 3, 0, 2, 4])
    np.random.shuffle(x_coordinate)
    col = 0
    observation_reflect = np.zeros((5, 3))
    while True:
        # 打开管道接收数据
        Final_clusters = [0, 0, 0]
        mediator_dict = queue.get()
        if mediator_dict:
            data_coord = mediator_dict["ball_deque_coord"]  # (x ,y ,w ,h ,depth ,code_color)
            clusters = np.array(data_coord)  # 转为array格式
            clusters = clusters[np.argsort(clusters[:, 1])]
            # 先将y进行排序，就是对第二列从小到大排序
            try:
                clusters_ = delete(total_get(clusters))[0]
                del Final_clusters[-len(clusters_):]
                Final_clusters[len(Final_clusters):] = clusters_
            except:
                logger.info("NO Ball")
            Final_clusters_ = np.array(Final_clusters)
            print(Final_clusters_)
            observation_reflect[col] = Final_clusters_
            env.reflect(observation_reflect.T)  # 映射到map上
            share_var['Reflect_img'] = env.map
            env.map_reset()

            action, min_x_best = model(x_coordinate, env.observation, color_code).main_decision()
            next_state, _, _ = env.step(action, 2)
            share_var['Best'] = action
            try:
                action2, min_x_better = model(x_coordinate, next_state, color_code).main_decision()
                next_state, _, _ = env.step(action2, 3)
                share_var['Better'] = action2
            except Exception:
                print("No other situation")


def Receiver(queue1):
    try:
        ret, ser2 = ser_open("/dev/ttyACM0")
        while ret:
            ser2.reset_input_buffer()  # 清除输入缓冲区
            recv = ser2.read(3)  # 读取数据并将数据存入recv
            print(recv)
            if recv[0] == 0xaa and recv[1] == 0xbb:
                queue1.put(recv[2])
    except:
        logger.info(f"Receiver ser is Died")


def Sender(queue2):
    try:
        ret, ser2 = ser_open("/dev/ttyUSB0")
        while ret:
            ser2.write(queue2.get())
    except:
        logger.info(f"Sender ser is Died")


def ser_open(port0="/dev/ttyUSB1"):
    ser2 = serial.Serial()
    ser2.baudrate = 115200  # 波特率
    ser2.port = port0
    ser2.open()
    if ser2.isOpen():
        print("串口打开成功")
    return ser2.isOpen(), ser2


class Track(object):
    # 继承plot的类,传入一个共享变量在并行处理的时候可以读取到BucketTime的各种信息
    def __init__(self):
        # todo 设立一个中介字典，存储并转换球，之后用一个聚类算法，将x作为聚类分为5簇，y则直接排序，并且当col到一定量的时候才会更新这个字典
        self.share_data = dict
        self.mediator_dict = {"ball_deque_coord": deque()}

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
        # 相机初始化画面为640^480
        realsense_cap = Realsense.realsense()
        cap = realsense_cap.cam_init(640)
        imgsz = check_img_size(imgsz, s=stride)  # check image size,确定推理照片尺寸
        imgs = [None]  # 齐次化做准备
        while True:
            self.mediator_dict["ball_deque_coord"] = deque()
            color_image, depth_colormap, depth_intrin, aligned_depth_frame = realsense_cap.cam_run(cap)
            # 将图片旋转90度进行判断
            imgs[0] = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
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
                        cv2.circle(color_image, (mid_pos[1], realsense_cap.height - mid_pos[0]), 1, colors(int(cls)),
                                   3)  # 圆心
                        # cv2.putText(im0, f"{mid_pos}", mid_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        # 由于识别的时候将画面转移了90°，为了对齐深度的像素，需要将像素逆变换回去，才能拿
                        target_width, target_height = abs(int(xyxy[2]) - int(xyxy[0])), abs(int(xyxy[3]) - int(xyxy[1]))
                        min_val = min(target_width, target_height)  # 通过识别框的大小确定一个深度范围
                        z, y, x = realsense_cap.depth_to_data(depth_intrin, aligned_depth_frame,
                                                              (mid_pos[-1], realsense_cap.height - mid_pos[-2]),
                                                              min_val)
                        cv2.putText(im0, f"{round(x, 2)}, {round(y, 2)}, {round(z, 2)}", mid_pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        if names[int(cls)] == "red":
                            self.mediator_dict["ball_deque_coord"].append(
                                [*mid_pos, target_width, target_height, round(z, 2), int(redcode)])
                        elif names[int(cls)] == "blue":
                            self.mediator_dict["ball_deque_coord"].append(
                                [*mid_pos, target_width, target_height, round(z, 2), int(bluecode)])
                        label = '%s %.2f' % (names[int(cls)], conf)
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))
                    queue.put(self.mediator_dict)
                im0 = annotator.result()
                self.share_data["Reflect_data"] = Reflect_data
                # 以下是向外输送
                self.share_data["im0"] = im0
                self.share_data["img_color"] = color_image

    # This function is used to choose argument,which is called init function
    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/bestBall.pt',
                            help='model path or triton URL')
        parser.add_argument('--source', type=str, default="0",
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
        global queue, Reflect_data
        queue = multiprocessing.Queue(maxsize=2)
        Reflect_data = multiprocessing.Manager().dict()
        # 初始化赋值["Best_ROI"],Reflect_data["Best_ROI"]避免先启动读不到数据
        # share_var['Best_track']
        Reflect_data["Best"] = None
        # 为了将数据送出，构建一个共享字典
        self.share_data = shareVar
        # 使其为全局变量，不然无法添加
        opt = self.parse_opt()
        check_requirements(exclude=('tensorboard', 'thop'))
        lib_init()
        # 启动进程
        p2 = multiprocessing.Process(target=FUCKING, args=(queue, Reflect_data), daemon=True)
        p2.start()
        # 便于关闭所有进程
        self.run_normal(**vars(opt))


if __name__ == "__main__":
    shareVar = multiprocessing.Manager().dict()
    t_1 = multiprocessing.Process(target=Track().main, args=(shareVar,))
    t_1.start()
    while True:
        try:
            cv2.imshow("1231", shareVar['im0'])
            # cv2.imshow('321', shareVar["img_color"])
            cv2.imshow('321', shareVar["Reflect_data"]["Reflect_img"])
            # print('次要选择', shareVar["Reflect_data"]["Better"])
            # print('最佳选择_x', shareVar["Best_x"])
            cv2.waitKey(1)
        except Exception:
            pass
