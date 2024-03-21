
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
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
import ydlidar

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


try:
    ser1 = serial.Serial()
    ser1.port = '/dev/ttyUSB0'
    ser1.baudrate = 115200
except Exception:
    pass

port = "/dev/ttyUSB0"; #串口号

# laser = ydlidar.CYdLidar();
# laser.setlidaropt(ydlidar.LidarPropSerialPort, port);
# laser.setlidaropt(ydlidar.LidarPropSerialBaudrate, 115200)
# laser.setlidaropt(ydlidar.LidarPropLidarType, ydlidar.TYPE_TRIANGLE);
# laser.setlidaropt(ydlidar.LidarPropDeviceType, ydlidar.YDLIDAR_TYPE_SERIAL);
# laser.setlidaropt(ydlidar.LidarPropScanFrequency, 10.0);
# laser.setlidaropt(ydlidar.LidarPropSampleRate, 6);
# laser.setlidaropt(ydlidar.LidarPropSingleChannel, True);
# laser.setlidaropt(ydlidar.LidarPropMaxAngle, 180.0); #设置扫描的max角度
# laser.setlidaropt(ydlidar.LidarPropMinAngle, -180.0); #设置扫描的min角度
# laser.setlidaropt(ydlidar.LidarPropMaxRange, 3.5 ); #设置最大识别距离
# laser.setlidaropt(ydlidar.LidarPropMinRange, 0.01);#min range dist
# scan = ydlidar.LaserScan()


def animate_lidar(radium):
    #lidar start
    laser.doProcessSimple(scan);
    dist=0
    for point in scan.points:
        if radium-2 < point.angle * 57.295 < radium+1:
            # print(point.range,"dist")
            # print(point.angle*57.295,"angle\n")
            dist=point.range
    return dist


def animate():
    distance=0
    # if count > 8:
    # read的长度是，收取数据的长度
    recv = ser1.read(9)  # 读取数据并将数据存入recv
    # print('get data from serial port:', recv)
    ser1.reset_input_buffer()  # 清除输入缓冲区
    if recv[0] == 0x59 and recv[1] == 0x59:  # python3
        distance = np.int16(recv[2] + np.int16(recv[3] << 8))
        # strength = recv[4] + recv[5] * 256
        # temp = (np.int16(recv[6] + np.int16(recv[7] << 8)))/8-256 #计算芯片温度
        distance = distance / 100
        # print('distance = %5d  strengh = %5d  temperature = %5d' % (distance, strength, temp))
        ser1.reset_input_buffer()
    # print(distance,"m")
    return distance



#角度畸变
def change(real_distance,angle):
    lengh=0.35252
    angle=angle/57.295779
    x=math.cos(angle)*(real_distance)
    y=math.sin(angle)*(real_distance)
    k=lengh+x
    angle=math.atan(y/k)*57.295779

    return angle



def open_ser():
    port1 = '/dev/ttyUSB1'
    port2='/dev/ttyUSB0'# 串口号
    baudrate = 115200  # 波特率
    try:
        global ser2
        ser2 = serial.Serial(port1, baudrate, timeout=0.1)
        if (ser2.isOpen() == True):
            print("串口打开成功")
    except Exception as exc:
        try:
            ser2=ser2 = serial.Serial(port2, baudrate, timeout=0.1)
            if (ser2.isOpen() == True):
                print("串口打开成功")
        except Exception as exc:
            print(exc)

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
def sjcl(a):
    # if a>=0:
    a = int(abs(a))
    a2="0"
    if a < 10:
        a2 = str(0) +str(0)+ str(a);
    elif 10 <= a < 150:
        a2 = str(0) + str(a);
    return a2





@smart_inference_mode()
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
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 配置depth流
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 配置color流
    pipeline.start(config)  # 流程开始
    align_to = rs.stream.color  # 与color流对齐
    align = rs.align(align_to)
    angle_listA=["A"]
    angle_listB=[]
    angle_listC=[]
    radium_listA=[]
    radium_listB=[]
    radium_listC=[]
    dist_steady=0
    state=0
    while True:
            # #获取相机参数
            frames = pipeline.wait_for_frames()  # 等待获取图像帧
            aligned_frames = align.process(frames)  # 获取对齐帧
            aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
            color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧
            ############### 相机参数的获取 #######################
            # intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
            depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
            color_image = np.asanyarray(color_frame.get_data())  # RGB图
            depth_frame =aligned_depth_frame
            mask = np.zeros([color_image.shape[0], color_image.shape[1]], dtype=np.uint8)
            mask[0:720, 320:1280] = 255
            #cv2,process

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

            im = torch.from_numpy(img).to(model.device)

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # Inference
            visualize =  False
            # print(im)
            pred = model(im, augment=augment, visualize=visualize)
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            for i, det in enumerate(pred):  # per image
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                s += '%gx%g ' % img.shape[2:]  # print string

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
                        # print("mid_pos = ",mid_pos)
                        min_val = min(abs(int(xyxy[2]) - int(xyxy[0])), abs(int(xyxy[3]) - int(xyxy[1])))  # 确定深度搜索范围
                        randnum = 40
                        for i in range(randnum):
                            bias = random.randint(-min_val // 8, min_val // 8)
                            dist = depth_frame.get_distance(int(mid_pos[0] + bias), int(mid_pos[1] + bias))
                            if dist:
                                distance_list.append(dist)
                        distance_list = np.array(distance_list)
                        distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]  # 冒泡排序+中值滤波
                        # if view_img:  # Add bbox to image
                        #颜色
                        c = int(cls)  # integer class
                        mean_distance=np.mean(distance_list)
                        #label 为 深度输出数值
                        #计算角度
                        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, mid_pos, mean_distance)
                        # print(camera_coordinate[0],"y")
                        # print(camera_coordinate[1],"z")
                        # print(camera_coordinate[2],"x")
                        #兔子的RGB摄像头与测距相对居中
                        y_cl=float(camera_coordinate[0])+0.020
                        z_cl=abs(float(camera_coordinate[1]))
                        # print(round(z_cl,2),":height")
                        #度制的转换
                        tan_nums=y_cl/mean_distance
                        #返回的是物体到雷达中心的距离
                        real_distance=math.sqrt(mean_distance**2+(y_cl)**2)
                        radium=math.atan(tan_nums)
                        radium=radium*57.295779
                        # print("nochange",radium)
                        #避免出现扫描不到出现的误区
                        if radium<1000:
                            #下列函数开启则，返回畸变的距离角度信息
                            radium = round(radium)
                            radium = change(real_distance, radium)
                            #中间角度是50
                            radium+=50
                            radium = int(radium)
                            # real_distance = animate_lidar(radium + 90)
                            x1=xyxy[0]
                            y1=xyxy[1]
                            x2=xyxy[2]
                            y2=xyxy[3]
                            scale=(x1-x2)/(y1-y2)
                            label = '%.2f %.2f%s%.2f%s' % (z_cl, real_distance, "m", radium, '.')
                            # label=f"{round(z_cl,2)}"
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            #little column
                            if z_cl<0.30:
                                radium_listA.append(radium)
                            #mid column
                            elif z_cl<0.50:
                                radium_listB.append(radium)
                            #large column
                            elif z_cl<1.00:
                                radium_listC.append(radium)
                            #角度的排序，便于传送数据更新抖动小
                    # dist_lidar = animate()
                    dist_lidar=0
                    radium_listA.sort()
                    radium_listC.sort()
                    radium_listB.sort()
                    # print(radium_listB)
                    for i in radium_listA:
                        # real_distance = animate_lidar(i +40)
                        # print(real_distance, 'm:Adist')
                        # rpm=str(int(real_distance*100))
                        radium_in = sjcl(i)
                        angle_listA.append(radium_in)
                        # angle_listA.append(rpm)
                    #长度取决于，需要检测柱子的多少
                    if len(angle_listA) < 10:
                        #确定输出柱子的数量n串口打开成功
                        # print(angle_list)
                        for i in range(3-len(radium_listA)):
                            angle_listA.append("000")


                    for i in radium_listB:
                        # print(i)
                        # real_distance = animate_lidar(i +40)
                        # rpm = str(int(real_distance*100))
                        radium_in = sjcl(i)
                        angle_listB.append(radium_in)
                        # angle_listA.append(rpm)
                    #长度取决于，需要检测柱子的多少
                    if len(angle_listB) < 10:
                        #确定输出柱子的数量n串口打开成功
                        # print(angle_list)
                        for i in range(4-len(radium_listB)):
                            angle_listB.append("000")



                    for i in radium_listC:
                        #获取雷达距离，从而计算rpm
                        # real_distance = animate_lidar(i +40)
                        # print(real_distance, ':Cdist')
                        # rpm = str(int(real_distance*100))
                        radium_in = sjcl(i)
                        angle_listC.append(radium_in)
                        # angle_listA.append(rpm)
                    #长度取决于，需要检测柱子的多少
                    if len(angle_listC) < 10:
                        #确定输出柱子的数量n串口打开成功
                        for i in range(1-len(radium_listC)):
                            angle_listC.append("000")
                    angle_all = ''.join(angle_listA+angle_listB+angle_listC)

                    if state>20:

                        #做一个滤波为，一定距离
                        if abs(dist_lidar-dist_steady)<0.05:
                            # dist=str(int(dist*100))
                            # print("当前距离为 = ", dist)
                            # angle_all+=dist
                            # print(angle_all)
                            print(dist_steady,"m")
                            print(angle_all)
                            print("a", angle_listA, '\n', "b", angle_listB, '\n', 'c', angle_listC, '\n')
                            state = 0
                            open_ser()
                            send_msg(angle_all)
                        else:
                            dist_steady=dist_lidar
                        #在数据中加上距离
                        # # dist=str(int(dist*100))
                        # # print("当前距离为 = ", dist)
                        # # angle_all+=dist
                        # # print(angle_all)
                        # print(angle_all)
                        # print("a",angle_listA,'\n',"b",angle_listB,'\n','c',angle_listC,'\n')
                        # state=0
                        # open_ser()
                        # send_msg(angle_all)
                    else:
                        state+=1
                    angle_listA = ["A"]
                    angle_listB = []
                    angle_listC = []
                    radium_listA = []
                    radium_listB = []
                    radium_listC = []

                else:
                    #reset
                    angle_list = ['A', '000', '000', '000', '000','000','000',"000","000"]
                    angle_all = ''.join(angle_list)
                    radium=''.join(angle_list)
                    print(angle_list)
                    print(angle_all)
                    open_ser()
                    send_msg(radium)

                # Stream results

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                # images_real = np.hstack((color_image, depth_colormap))
                im0 = annotator.result()
                # if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

                cv2.imshow(str(p), im0)
                # cv2.imshow("real",images_real)
                cv2.waitKey(1)  # 1 millisecond



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/best2.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / '0', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=10, help='maximum detections per image')
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
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
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
    # try:

    while True:
        try:
            ser1.open()
        except Exception:
            pass
        # try:
        #     laser.initialize();
        #     laser.turnOn();
            opt = parse_opt()
            main(opt)
        # except Exception:
        #     pass





#矮的
# x=float(input("dist="))
# y=3756.9018186759827+-130.92056071064977*x+(2.8899168388705085)*x**2+(-0.036433581123053574)*x**3+(0.0002925050320336321)*x**4+(-1.5669926472399643e-06)*x**5+(5.681503807744026e-09)*x**6+(-1.377522419000044e-11)*x**7+(2.137859685607119e-14)*x**8+(-1.9180089521971135e-17)*x**9+(7.556155754291958e-21)*x**10
# print(round(y))


#高
# x=float(input("dist="))
# y=-11135.16888603246*x**0+496.343628705211*x+(-7.262556232949655)*x**2+(0.034447810118269716)*x**3+(0.0002570270128829506)*x**4+(-4.269691345092778e-06)*x**5+(2.220673825220269e-08,)*x**6+(-3.673604757202774e-11)*x**7+(-1.1004216521085259e-13)*x**8+(4.719575661375775e-16)*x**9+(4.0346074957496114e-19)*x**10+(-5.268803815339255e-21)*x**11+(1.0844542344154707e-23)*x**12+(-5.8976458218509875e-27)*x**13+(-6.69433915420596e-30)*x**14+(7.292114505697707e-33)*x**15
# print(round(y))



 # [8296.444283993684, -328.14537763965046, 6.51673194768433, -0.07309275404987439, 0.0005243800914390464, -2.6838273245480532e-06, 1.1084962126733613e-08, -3.792089407224425e-11, 8.528465728480571e-14, -5.038831066631119e-17, -1.0376344707482638e-20, -2.234481984010945e-21, 1.2995514359452777e-23, -3.1956622223277587e-26, 3.8385182759886183e-29, -1.8572060906311345e-32]
 # [-11135.16888603246, 496.343628705211, -7.262556232949655, 0.034447810118269716, 0.0002570270128829506, -4.269691345092778e-06, 2.220673825220269e-08, -3.673604757202774e-11, -1.1004216521085259e-13, 4.719575661375775e-16, 4.0346074957496114e-19, -5.268803815339255e-21, 1.0844542344154707e-23, -5.8976458218509875e-27, -6.69433915420596e-30, 7.292114505697707e-33]