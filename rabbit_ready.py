# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:    `
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import math
import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import pyrealsense2 as rs
from numpy import random
import torch
import serial

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode



## 激光单点测距
# 设置雷达的波特率
def animate():
    # new detect dist
    distance = 0
    # recv = ser1.read(9)
    recv = ser1.read(9)
    ser1.reset_input_buffer()
    try:
        distance = float(eval((recv[2:7])))
    except Exception:
        pass

    print(distance)
    if distance < 25.500:
        pass
    else:
        distance = 0
    return distance


def open_ser(choice):
    global ser2
    port0 = '/dev/ttyUSB0'
    port1 = '/dev/ttyUSB1'
    ser2 = serial.Serial()
    ser2.baudrate = 115200  # 波特率
    try:
        if choice == 1:
            ser2.port = port0
            ser2.open()
        else:
            ser2.port = port1
            ser2.open()

        if (ser2.isOpen() == True):
            print("串口打开成功")
    except Exception:
        pass


# 数据发送
def send_msg(jd):
    try:
        ser2.write(str(jd).encode("gbk"))
    except Exception as exc:
        print("发送异常", exc)


# 关闭串口
def close_ser():
    try:
        ser2.close()
        if ser2.isOpen():
            print(" ")
        else:
            print(" ")
    except Exception as exc:
        print("串口关闭异常", exc)

def dis_sjcl(dist):
    if dist < 100:
        dist_n = str(0) + str(dist)
    else:
        dist_n = str(dist)
    return dist_n

def sjcl(h, a, dist):
    h = abs(h)
    # print(h)
    print(a)
    a2 = "A00001120"
    dist = int(dist)
    # mid
    # rpm2=8314.444283993684+(-328.14537763965046*dist**1)+(6.51673194768433*dist**2)+(-0.07309275404987439*dist**3)+(0.0005243800914390464*dist**4)+(-2.6838273245480532e-06*dist**5)+(1.1084962126733613e-08*dist**6)+(-3.792089407224425e-11*dist**7)+(8.528465728480571e-14*dist**8)+(-5.038831066631119e-17*dist**9)+(-1.0376344707482638e-20*dist**10)
    rpm2 = -598.2145131951667 + 52.21541408591279 * dist - 0.2584535640620177 * dist ** 2 - 0.007560598823961773 * dist ** 3 + 0.00014267092063554904 * dist ** 4 - 1.1697368633789746e-06 * dist ** 5 + 5.577822405750432e-09 * dist ** 6 - 1.644932880131149e-11 * dist ** 7 + 2.9634863472870304e-14 * dist ** 8 - 2.9955624715879514e-17 * dist ** 9 + 1.3037017918242376e-20 * dist ** 10

    rpm2 = int(round(rpm2))

    # min
    # rpm1=7831.751835502562+(-192.4798386841649*dist**1)+(2.210870397937463*dist**2)+(-0.012508056158068508*dist**3)+(3.4953085691955406e-05*dist**4)+(-3.862224621701064e-08*dist**5)
    # rpm1=158472.62955162593+(-1968.257652960921*dist**1)+(8.920403235096897*dist**2)+(-0.014487105087274948*dist**3)+(-8.545566094162918e-06*dist**4)+(3.9328060851448384e-08*dist**5)+(3.7417777694334365e-12*dist**6)+(3.22166149718354e-14*dist**7)+(-3.0692493445818155e-16*dist**8)+(3.0607075785850266e-19*dist**9)+(-4.545897880254303e-22*dist**10)+(1.769029430139664e-24*dist**11)+(-2.17146151944098e-28*dist**12)+(-6.764928048475142e-30*dist**13)+(9.615747488022059e-33*dist**14)+(-4.064465591268364e-36*dist**15)
    rpm1 = -679.9464244681135 + 78.00438562139381 * dist ** 1 + (
            -1.3068972896360933 * dist ** 2) + 0.011089218249408367 * dist ** 3 - 4.654341031140767e-05 * dist ** 4 + 5.0902717346576575e-08 * dist ** 5 + 3.0279166807541427e-10 * dist ** 6 - 8.54171323911783e-13 * dist ** 7 - 1.5011503884453209e-15 * dist ** 8 + 7.934952930187383e-18 * dist ** 9 - 7.376823164322936e-22 * dist ** 10 - 3.041923107450503e-23 * dist ** 11 + 9.778338532221371e-27 * dist ** 12 + 1.2649475933656475e-28 * dist ** 13 + -2.1258425789553487e-31 * dist ** 14 + 1.0649575929976705e-34 * dist ** 15

    rpm1 = int(round(rpm1))
    # max
    # rpm3=5216850.791654796+(-76762.08617478728*dist**1)+(349.75103636588045*dist**2)+(-0.10872160915296562*dist**3)+(-0.0023633265051444825*dist**4)+(-1.9866030827776386e-06*dist**5)+(2.830860434462129e-08*dist**6)+(-2.1367827014565922e-11*dist**7)+(9.463227997843033e-15*dist**8)+(-3.7785171495047655e-16*dist**9)+(1.0840045485961784e-18*dist**10)+(-3.005260302025057e-21*dist**11)+(5.588848178505915e-24*dist**12)+(1.2711988304402997e-26*dist**13)+(-5.668704350096854e-29*dist**14)+(5.09616962483101e-32*dist**15)
    rpm3 = -123555.42766620855 + 2596.8926424283336 * dist - 21.376771138055148 * dist ** 2 + 0.0821090655275402 * dist ** 3 - 8.626949751688833e-05 * dist ** 4 - 5.146096250611399e-07 * dist ** 5 + 2.578591026898844e-09 * dist ** 6 - 6.160318930920659e-12 * dist ** 7 + 9.726235529470033e-15 * dist ** 8 - 1.0069752048208477e-17 * dist ** 9 + 5.019127149964281e-21 * dist ** 10

    rpm3 = int(round(rpm3))
    #转速范围
    if 0<rpm3 < 1600:
        pass
    else:
        rpm3 = 1140
    if 0 < rpm1 < 1600:
        pass
    else:
        rpm1 = 1140
    if 0 < rpm2 < 1600:
        pass
    else:
        rpm2 = 1140

    if a >= 0:
        a = int(abs(a))
        #注意修改相关column height
        if h < 0.15:
            print("rpm1",rpm1)
            if a < 10:
                a2 = str("A") + str(0) + str(0) + str(a) + str(0) + str(rpm1);
            elif 10 <= a < 100:
                a2 = str("A") + str(0) + str(a) + str(0) + str(rpm1);
            elif a >= 100:
                a2 = str("A") + str(a) + str(0) + str(rpm1);
        elif h < 0.30:
            print("rpm2",rpm2)
            if a < 10:
                a2 = str("A") + str(0) + str(0) + str(a) + str(0) + str(rpm2);
            elif 10 <= a < 100:
                a2 = str("A") + str(0) + str(a) + str(0) + str(rpm2);
            elif a >= 100:
                a2 = str("A") + str(a) + str(0) + str(rpm2);
        elif h < 1.1:
            print("rpm3",rpm3)
            if a < 10:
                a2 = str("A") + str(0) + str(0) + str(a) + str(0) + str(rpm3);
            elif 10 <= a < 100:
                a2 = str("A") + str(0) + str(a) + str(0) + str(rpm3);
            elif a >= 100:
                a2 = str("A") + str(a) + str(0) + str(rpm3);
            # 输出的是cm
    else:
        a = int(abs(a))
        if h < 0.15:
            print("rpm1",rpm1)
            if a < 10:
                a2 = str("A") + str(0) + str(0) + str(a) + str(1) + str(rpm1);
            elif 10 <= a < 100:
                a2 = str("A") + str(0) + str(a) + str(1) + str(rpm1);
            elif a >= 100:
                a2 = str("A") + str(a) + str(1) + str(rpm1);
        elif h < 0.30:
            print("rpm2",rpm2)
            print("2")
            if a < 10:
                a2 = str("A") + str(0) + str(0) + str(a) + str(1) + str(rpm2);
            elif 10 <= a < 100:
                a2 = str("A") + str(0) + str(a) + str(1) + str(rpm2);
            elif a >= 100:
                a2 = str("A") + str(a) + str(1) + str(rpm2);
        elif h < 1.1:
            print("rpm3",rpm3)
            if a < 10:
                a2 = str("A") + str(0) + str(0) + str(a) + str(1) + str(rpm3);
            elif 10 <= a < 100:
                a2 = str("A") + str(0) + str(a) + str(1) + str(rpm3);
            elif a >= 100:
                a2 = str("A") + str(a) + str(1) + str(rpm3);

    return a2


# 修改图像的对比度,coefficent>0, <1降低对比度,>1提升对比度 建议0-2
def change_contrast(img, coefficent):
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
def change_brightness(img, brightness):
    [averB, averG, averR] = np.array(cv2.mean(img))[:-1] / 3
    k = np.ones((img.shape))
    k[:, :, 0] *= averB
    k[:, :, 1] *= averG
    k[:, :, 2] *= averR
    img = img + (brightness - 1) * k
    img[img > 255] = 255
    img[img < 0] = 0
    return img.astype(np.uint8)


def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    # model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    pipeline = rs.pipeline()  # 定义流程pipeline
    config = rs.config()  # 定义配置config
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # 配置depth流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 配置color流
    pipeline.start(config)  # 流程开始
    align_to = rs.stream.color  # 与color流对齐
    align = rs.align(align_to)
    radium_need = 0
    height = 0
    distance_steady = 0
    step = 0
    while True:
        # #获取相机参数
        radium_real = 135
        frames = pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = align.process(frames)  # 获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
        ############### 相机参数的获取 #######################
        # intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
        color_image = np.asanyarray(color_frame.get_data())  # RGB图
        # color_image=change_contrast(color_image,4)#对比度

        # color_image=change_brightness(color_image,3)#增亮
        # color_image=change_contrast(color_image,2)

        # color_image = cv2.GaussianBlur(color_image, (5, 5), 7)  # 高斯模糊API(模糊 可改参数)高斯核函数必须是奇数
        depth_frame = aligned_depth_frame
        mask = np.zeros([color_image.shape[0], color_image.shape[1]], dtype=np.uint8)
        mask[480:720, 640:1280] = 255
        # cv2,process

        sources = [source]
        imgs = [None]
        path = sources
        imgs[0] = color_image
        im0s = imgs.copy()
        img = [letterbox(x, new_shape=imgsz)[0] for x in im0s]
        img = np.stack(img, 0)
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to 3x416x416, uint8 to float32
        img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # img=change_contrast(img,2)
        im = torch.from_numpy(img).to(model.device)

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        visualize = False
        # print(im)ser1
        pred = model(im, augment=augment, visualize=visualize)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):  # per image
            p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print resultssave
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    distance_list = []
                    mid_pos = [int((int(xyxy[0]) + int(xyxy[2])) / 2),
                               int((int(xyxy[1]) + int(xyxy[3])) / 2)]  # 确定索引深度的中心像素位置左上角和右下角相加在/2

                    min_val = min(abs(int(xyxy[2]) - int(xyxy[0])), abs(int(xyxy[3]) - int(xyxy[1])))  # 确定深度搜索范围
                    # print(box,)
                    randnum = 80
                    for i in range(randnum):
                        bias = random.randint(-min_val // 20, min_val // 20)
                        dist = depth_frame.get_distance(int(mid_pos[0] + bias), int(mid_pos[1] + bias))

                        # print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
                        if dist:
                            distance_list.append(dist)
                    distance_list = np.array(distance_list)
                    distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]
                    # if view_img:  # Add bbox to image
                    # print(distance_list)
                    c = int(cls)  # integer class
                    mean_distance = np.mean(distance_list)
                    # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    # label 为 深度输出数值

                    # 计算角度
                    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, mid_pos, mean_distance)
                    # print(camera_coordinate[0],"y")
                    # print(camera_coordinate[1],"z")
                    # print(camera_coordinate[2],"x")
                    # right is +,left is -
                    y_cl = float(camera_coordinate[0]) + 0.07

                    # print(y_cl)sjcl
                    # 判断柱子高度，分辨柱子
                    z_cl = float(camera_coordinate[1])
                    # print(y_cl)
                    # 度制的转换
                    tan_nums = y_cl / (mean_distance + 0.15)
                    radium = math.atan(tan_nums)
                    radium = radium * 57.295779
                    # real_distance = math.sqrt(mean_distance ** 2 + y_cl ** 2)
                    # label = '%.2f%.2f%s%.z_cl1f%s' % (names[int(cls)], real_distance, 'm', radium, '.')
                    label = '%.1f %s %.2f%s%.3f%s' % (radium, 'm', mean_distance, 'm', abs(z_cl), 'm')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # 为了输出最中心的角度
                    if abs(radium) < radium_real:
                        radium_real = abs(radium)
                        radium_need = radium
                        height = z_cl
                        # need_distance=real_distance
                if radium_real < 135:
                    if step > 3:
                        radium_real = round(radium_need)
                        radium_real = int(radium_real)
                        # real
                        # need_distance = animate()
                        need_distance=0

                        # animate()
                        # print(distance_steady-need_distance)
                        # 滤波过滤变化低于4cm的变化
                        if abs(distance_steady - need_distance) > 0.01:
                            distance_steady = need_distance
                        # 将dist四舍五入化，稳定距离
                        distance_steady = round(distance_steady * 100)
                        # distance_steady = int(distance_steady)
                        # 前四位是角度，后三位是距离
                        radium_in = sjcl(height, radium_real, distance_steady)
                        radium_in += dis_sjcl(distance_steady)
                        # print(radium_real)
                        print(radium_in)
                        print("激光=", distance_steady, 'cm')
                        # open_ser(choice=choice)
                        # send_msg(radium_in)
                        step = 0
                    step += 1
                else:
                    radium = str("A") + str("00001120000")
                    # open_ser(choice=choice)
                    # send_msg(radium)

            else:
                radium = str("A") + str("00001120000")
                # open_ser(choice=choice)
                # send_msg(radium)

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images_real = np.hstack((color_image, depth_colormap))
            im0 = annotator.result()
            # if view_img:
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

            cv2.imshow(str(p), im0)
            cv2.imshow("real",images_real)
            cv2.waitKey(1)  # 1 millisecond


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/best.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / '0', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1080], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=100, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    # ser1 = serial.Serial()
    # ser1.port = '/dev/ttyUSB0'  # 设置端口
    # ser1.baudrate = 38400
    # ser1.timeout = 0.2
    # ser1.open()
    while True:
        # recv = ser1.read(9)
        # ser1.reset_input_buffer()
        # if recv:
        #     choice = 0
        #     distance = float(eval((recv[2:7])))
        #     ser1.reset_input_buffer()
        # else:
        #     choice = 1
        #     ser1.close()
        #     ser1.port = '/dev/ttyUSB1'
        #     ser1.open()
        #     ser1.reset_input_buffer()
        opt = parse_opt()
        main(opt)
        # except Exception:
        # pass
