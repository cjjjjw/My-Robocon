import math
import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np

from Map_Reflect_utils import Realsense
from numpy import random
import torch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing
import time
import serial

pi = math.pi

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


def process1(share_var):
    # 创建绘制实时损失的动态窗口
    plt.ion()
    fig = plt.figure(dpi=100)
    ax = Axes3D(fig)
    # 创建循环
    while True:
        ax.clear()
        ax.set_zlabel('Z', fontdict={'size': 10, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 10, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 10, 'color': 'red'})
        ax.set_xlim(0, 5)
        ax.set_ylim(3, -3)
        ax.set_zlim(0, 5)
        ax.scatter(0, 0, 0.3, c='g')
        ax.text(0, 0, 0.3, "center")
        if len(share_var) > 0:
            for i in range(len(share_var)):
                ax.scatter(share_var[i][0], share_var[i][1], share_var[i][2], c=f"{share_var[i][3]}")
                ax.text(share_var[i][0], share_var[i][1], share_var[i][2], f"{share_var[i][3]}")
            del share_var[:]
        plt.pause(0.003)
        plt.ioff()


def open_ser() -> bool:
    global ser2
    port0 = 'COM13'
    ser2 = serial.Serial()
    ser2.baudrate = 115200  # 波特率
    try:
        ser2.port = port0
        ser2.open()
        if ser2.isOpen() == True:
            print("串口打开成功")
        return True
    except Exception:
        return False
        pass


def send_msg(jd):
    try:
        ser2.write(str(jd).encode("gbk"))
    except Exception as exc:
        print("发送异常", exc)


def send_data_change(data: float) -> str:
    data = round(data)
    data_postive = abs(data)
    if data > 0:
        data_ = str("P")
    else:
        data_ = str("N")
    if data_postive >= 1000:
        data_ += str(data_postive)
    elif data_postive >= 100:
        data_ += str("0")+str(data_postive)
    elif data_postive >= 10:
        data_ += str("00") + str(data_postive)
    elif data_postive >= 0:
        data_ += str("000") + str(data_postive)
    return data_


def data_solve(best_coord, dist_):
    # 计算x，y的点和距离
    str_send = str('kk') + send_data_change(best_coord[0] * 1000) + send_data_change(
        best_coord[1] * 1000) + send_data_change(dist_ * 10)
    print(str_send)
    send_msg(str_send)


class track(object):
    def __init__(self):
        self.realsense_cap = Realsense.realsense()
        self.depth_intrin = None
        self.aligned_depth_frame = None

    # 有些库无法提前引用
    def lib_init(self):
        global DetectMultiBackend
        global Annotator, colors, save_one_box
        from models.common import DetectMultiBackend
        # # box draw
        from utils.plots import Annotator, colors, save_one_box

    def laser(self):
        # todo 为了纠正imu的累积误差
        pass

    # 计算最短距离
    def calc(self, x, y, z, center):
        return math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)

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

    def image_stream(self, classes="source_img"):
        global cap
        if classes == "0":
            for i in range(10):
                cap = cv2.VideoCapture(1)
                if cap.isOpened():
                    break
        elif classes == "1":  # realsense
            align = self.realsense_cap.cam_init(640)  # 引用相机,初始化相机
            return align  # 返回时，退出函数
        else:
            cap = cv2.VideoCapture(classes)
        return cap  # 返回打开的设备参数

    def camera_to_world(self, xc, yc, zc):
        # xc,yc,zc是相机坐标系下的坐标，现在转化为世界坐标系下的坐标
        RT = self.pose_robot(0, 120, 0, 0, 0, 0.30)
        trans = np.array([xc, yc, zc, 1]).T
        x_tran, y_tran, z_tran, _ = (RT @ trans)  # 传出作为全局变量s
        return x_tran, y_tran, z_tran

    def myRPY2R_robot(self, x, y, z):
        Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return R

    # 用于根据位姿计算变换矩阵
    def pose_robot(self, x, y, z, Tx, Ty, Tz):
        thetaX = x / 180 * pi
        thetaY = y / 180 * pi
        thetaZ = z / 180 * pi
        R = self.myRPY2R_robot(thetaX, thetaY, thetaZ)
        t = np.array([[Tx], [Ty], [Tz]])
        RT1 = np.column_stack([R, t])  # 列合并
        RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
        # RT1=np.linalg.inv(RT1)
        return RT1

    def run_3d(self, weights=ROOT / 'yolov5s.pt',  # model path or triton URL
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
        global data_things, dist_last, coord_3d
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # fp16半精度模型，dnn 加速推理
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size,确定推理照片尺寸
        imgs = [None]  # 齐次化做准备
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        cap = self.image_stream("1")
        ser_flag = open_ser()
        # 声明一个列表进程变量
        while True:
            min_dist = 1000
            imgs[0], image_depth, self.depth_intrin, self.aligned_depth_frame = self.realsense_cap.cam_run(cap)
            imgs[0] = self.change_brightness(imgs[0], 1)
            # copy another image
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
                coord = (int(im0.shape[1] / 2), im0.shape[0])
                s += '%gx%g ' % img.shape[2:]  # print string
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):  # 判读是否有物体识别到
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # 坐标映射
                    # Print resultssave
                    for c in det[:, 5].unique():  # 识别物的类别
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    for *xyxy, conf, cls in reversed(det):  # 识别物的类别像素坐标，置信度，物体类别号
                        mid_pos = [int((int(xyxy[0]) + int(xyxy[2])) / 2),
                                   int((int(xyxy[1]) + int(xyxy[3])) / 2)]
                        cv2.circle(im0, mid_pos, 2, (0, 255, 0), 2)
                        min_val = min(abs(int(xyxy[2]) - int(xyxy[0])), abs(int(xyxy[3]) - int(xyxy[1])))
                        # 将中心点向四周扩展计算深度
                        z, y, x = self.realsense_cap.depth_to_data(self.depth_intrin, self.aligned_depth_frame, mid_pos,
                                                                   min_val)
                        # 换到车上坐标系
                        x, y, z = self.camera_to_world(x, y, z)
                        x, y, z = round(x, 4), round(y, 4), round(z, 4)
                        label = 'x=%.2f,y=%.2f,z=%.2f,%s' % (x, y, z, names[int(cls)])
                        # 添加目标物参数
                        dist_last = self.calc(x, y, z, (0, 0, 0))
                        if min_dist > dist_last:
                            min_dist = dist_last
                            coord = mid_pos
                            coord_3d = (x, y, z)
                        data_things.append((x, y, z, names[int(cls)]))
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))
                if ser_flag:
                    if len(det):
                        data_solve(coord_3d, round(dist_last * 100, 2))
                    else:
                        data_solve((0, 0, 0), 0)
                    im0 = annotator.result()
                cv2.line(im0, (int(im0.shape[1] / 2), im0.shape[0]), coord, (255, 0, 0), 3)
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # view_img
                cv2.imshow(str(1), im0)
                if (cv2.waitKey(1)) == 'q':  # 1 millisecond
                    break

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
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size,确定推理照片尺寸
        imgs = [None]  # 齐次化做准备
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        cap = self.image_stream(source)
        while True:
            ret, imgs[0] = cap.read()
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
                    for c in det[:, 5].unique():  # 识别物的类别
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    for *xyxy, conf, cls in reversed(det):  # 识别物的类别像素坐标，置信度，物体类别号
                        mid_pos = [int((int(xyxy[0]) + int(xyxy[2])) / 2),
                                   int((int(xyxy[1]) + int(xyxy[3])) / 2)]
                        cv2.circle(im0, mid_pos, 1, colors(int(cls)), 3)  # 圆心
                        label = '%s' % (s)
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))
                im0 = annotator.result()
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # view_img
                cv2.imshow(str(1), im0)
                cv2.waitKey(1)  # 1 millisecond

    # This function is used to choose argument,which is called init function
    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/best.pt',
                            help='model path or triton URL')
        parser.add_argument('--source', type=str, default='0',
                            help='')
        parser.add_argument('--data', type=str, default=ROOT / 'data/ball.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                            help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.55, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=100, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--classes', default=1, nargs='+', type=int,
                            help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', default=True, action='store_true', help='use OpenCV DNN for ONNX inference')
        parser.add_argument('--camera_device', type=str, default="1",
                            help='use which 0=local camera and root flie,1=Realsense,2=OAK')

        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt

    # start function
    def main(self):
        # 使其为全局变量，不然无法添加
        global data_things
        opt = self.parse_opt()
        check_requirements(exclude=('tensorboard', 'thop'))
        if vars(opt)['camera_device'] == "1":
            data_things = multiprocessing.Manager().list()
            # 设置进程锁
            new_process = multiprocessing.Process(target=process1, args=(data_things,))
            new_process.start()
            # 先启动可视化进程，再打开主进程,避免阻塞，与主线程共享变量
            self.lib_init()
            del vars(opt)['camera_device']
            self.run_3d(**vars(opt))  # 解引用
        else:
            self.lib_init()
            del vars(opt)['camera_device']
            self.run_normal(**vars(opt))


if __name__ == "__main__":
    track = track()
    track.main()

    # track.main()
