import argparse
import numpy as np
import cv2
import time
from utils_track import vino as ov
from collections import deque
import multiprocessing
from Map_Reflect_utils import if_model, Map_Reflect, Realsense
from numba import jit
import logging
import serial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("Depth and Reflect")

# 定义棋子
bluecode = 1.0
redcode = -1.0
backcode = 0.0


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


def ser_open(port0="/dev/ttyUSB2"):
    ser2 = serial.Serial()
    ser2.baudrate = 115200  # 波特率
    ser2.port = port0
    if ser2.isOpen():
        print("串口打开成功")
    else:
        ser2.open()
    return ser2.isOpen(), ser2


def Receiver(queue1):
    try:
        ret, ser2 = ser_open("/dev/ttyACM0")
        while ret:
            ser2.reset_input_buffer()  # 清除输入缓冲区
            recv = ser2.read(3)  # 读取数据并将数据存入recv
            # print(recv)
            if recv[0] == 0xaa and recv[1] == 0xbb:
                queue1.put(recv[2])
    except:
        logger.info(f"Receiver ser is Died")


def Sender(queue2):
    # 提前设立好该发送的信号，当检测到状态跳变，在改变

    try:
        ret, ser1 = ser_open("/dev/ttyUSB0")
        while ret:
            ser1.reset_input_buffer()  # 清除输入缓冲区
            # print(queue2.get())
            ser1.write(queue2.get())
    except:
        logger.info(f"Sender ser is Died")


def FUCKING(share_var, queue, code, Receiver_queue, Sender_queue):
    # 先进行按列排序
    env = Map_Reflect.State()
    env.map_init(False)  # 地图启动
    model = if_model.Judgment
    share_var['Reflect_img'] = env.map
    x_coordinate = np.array([5, 3, 0, 2, 4])
    np.random.shuffle(x_coordinate)
    observation_reflect = np.zeros((5, 3))
    # 打开管道接收数据
    last_ser = 0
    last_best = None
    # 设置一个detect_flag，表明是否已经是检测框内的情况
    now_detect_flag = False
    input_detect_flag = False
    while True:
        # 打开管道接收数据，初始化
        Final_clusters = [0, 0, 0]
        # 先做第一个球的位置计算
        action_best, min_x_best = model(x_coordinate, env.observation, code, False).main_decision()
        next_state, _, _ = env.step(action_best, 2)
        try:
            action_better, min_x_better = model(x_coordinate, next_state, code, False).main_decision()
            next_state, _, _ = env.step(action_better, 3)
            share_var['Better'] = action_better
        except Exception:
            print("No other situation")
        share_var['Best'] = action_best
        share_var['Reflect_img'] = env.map
        env.map_reset()  # 清除图表，避免溢出
        # 先传输移动命令到，下位机ser.send(best_action)

        print(now_detect_flag, " now_target")
        print(input_detect_flag, "last_target")
        set_flag = Receiver_queue.get()
        if not now_detect_flag and input_detect_flag and set_flag == 1:
            Sender_queue.put(str(f"5A{action_best}{1}00000000000000").encode("gbk"))
        else:
            print(str(f"5A{action_best}{0}00000000000000"))
            Sender_queue.put(str(f"5A{action_best}{0}00000000000000").encode("gbk"))

        # print(set_flag, " send")
        # print(last_ser, "ser")
        # 当发生框检验，则发生一次状态记录
        if now_detect_flag:
            if last_best == action_best:
                input_detect_flag = True
                # 释放放球的命令
                upgrade_state, _, _ = env.step(action_best, code)
                env.reflect(upgrade_state)
                observation_reflect = upgrade_state.T

        # if not queue.empty():
        now_detect_flag = False  # 框检验位，重新设置为0
        mediator_dict = queue.get()
        data_coord = mediator_dict["ball_deque_coord"]  # (x ,y ,w ,h ,depth ,code_color)
        clusters = np.array(data_coord)  # 转为array格式
        try:
            clusters = clusters[np.argsort(clusters[:, 1])]  # 按照像素Y来进行排序
            clusters_ = delete(total_get(clusters))[0]
            del Final_clusters[-len(clusters_):]
            Final_clusters[len(Final_clusters):] = clusters_
        except:
            logger.info("NO Ball")
        # 一直更新最佳位置，但是却不是释放放球的命令
        # 当发生反转信号，set_flag==1，last_ser==0，则发生一次状态记录
        if set_flag == 1 and last_ser != 1:
            # 当确认到达后，才开始计算是否放入
            Final_clusters_ = np.array(Final_clusters)
            observation_reflect[action_best] = Final_clusters_
            last_best = action_best
            now_detect_flag, input_detect_flag = True, False

        env.reflect(observation_reflect.T)  # 映射到map上
        # 发生反转信号前，都与set_flag 相同
        last_ser = set_flag


class Track(object):
    # 继承plot的类,传入一个共享变量在并行处理的时候可以读取到BucketTime的各种信息
    def __init__(self):
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

    def run_normal(self, model):
        realsense_cap = Realsense.realsense(1)
        cap = realsense_cap.cam_init(640)
        # 打开管道接收数据
        while True:
            self.mediator_dict["ball_deque_coord"] = deque()
            color_image, depth_colormap, depth_intrin, aligned_depth_frame = realsense_cap.cam_run(cap)
            # 将图片旋转90度进行判断
            frame = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
            # copy to become another image
            # 当接收到到达的信息时才开始，进行识别
            if not Receiver_queue.empty():
                set_flag = Receiver_queue.get()
                if set_flag == 1:  # 当串口标志位发生变化的时候，才开始进行识别
                    frame, det = model.run(frame)
                    if len(det):
                        for xyxy, conf, cls in reversed(det):  # 识别物的类别像素坐标，置信度，物体类别号
                            mid_pos = [int((xyxy[0] + xyxy[2]) / 2),
                                       int((xyxy[1] + xyxy[3]) / 2)]
                            cv2.circle(frame, mid_pos, 1, (255, 0, 0), 3)  # 圆心
                            # cv2.putText(im0, f"{mid_pos}", mid_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            # 由于识别的时候将画面转移了90°，为了对齐深度的像素，需要将像素逆变换回去，才能拿
                            target_width, target_height = abs(int(xyxy[2] - xyxy[0])), abs(int(xyxy[3] - xyxy[1]))
                            min_val = min(target_width, target_height)  # 通过识别框的大小确定一个深度范围
                            z, y, x = realsense_cap.depth_to_data(depth_intrin, aligned_depth_frame,
                                                                  (mid_pos[-1], realsense_cap.height - mid_pos[-2]),
                                                                  min_val)
                            cv2.putText(frame, f"{round(x, 2)}, {round(y, 2)}, {round(z, 2)}", mid_pos,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            if cls == 0:
                                self.mediator_dict["ball_deque_coord"].append(
                                    [*mid_pos, target_width, target_height, round(z, 2), int(redcode)])
                            elif cls == 1:
                                self.mediator_dict["ball_deque_coord"].append(
                                    [*mid_pos, target_width, target_height, round(z, 2), int(bluecode)])
            queue.put(self.mediator_dict)
            # 以下是向外输送
            self.share_data["Reflect_data"] = Reflect_data
            self.share_data["im0"] = frame

    # This function is used to choose argument,which is called init function
    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_path', type=str,
                            default="/home/nuc2/PycharmProjects/yolov5-master/weights/bestBallll.xml",
                            help='model path')
        parser.add_argument('--weights_path', nargs='+', type=str,
                            default="/home/nuc2/PycharmProjects/yolov5-master/weights/bestBallll.bin",
                            help='weights path or triton URL')
        parser.add_argument('--conf-thres', type=float, default=0.8, help='confidence threshold')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--iou-thres', type=float, default=0.55, help='NMS IoU threshold')
        parser.add_argument('--classes', type=list, default=[], help='Classes')
        parser.add_argument('--img-size', type=int, default=736, help='img-size')
        parser.add_argument('--device', default='GPU', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--name-porcess', default='BucketTime', help='name to processs--cjh')
        parser.add_argument('--source', type=str, default="0",
                            help='')
        opt = parser.parse_args()

        return opt

    # start function
    def main(self, shareVar, code=bluecode):
        global queue, Reflect_data, Receiver_queue
        Reflect_data = multiprocessing.Manager().dict()
        # 初始化赋值["Best_ROI"],Reflect_data["Best_ROI"]避免先启动读不到数据
        # share_var['Best_track']
        queue = multiprocessing.Manager().Queue(maxsize=4)
        Reflect_data["Best"], Reflect_data["Better"], Reflect_data["ball_deque_coord"] = None, None, deque()
        Receiver_queue = multiprocessing.Manager().Queue(maxsize=4)
        Sender_queue = multiprocessing.Manager().Queue(maxsize=4)
        # 为了将数据送出，构建一个共享字典
        self.share_data = shareVar
        # 使其为全局变量，不然无法添加
        opt = self.parse_opt()
        del vars(opt)["source"]
        model = ov.Vino(**vars(opt))
        # 启动进程
        p1 = multiprocessing.Process(target=FUCKING, args=(Reflect_data, queue, code, Receiver_queue, Sender_queue))
        p2 = multiprocessing.Process(target=Receiver, args=(Receiver_queue,))
        p3 = multiprocessing.Process(target=Sender, args=(Sender_queue,))
        p2.start()
        p3.start()
        p1.start()
        self.share_data["pid"] = [p1.pid, p2.pid, p3.pid]
        # 便于关闭所有进程
        self.run_normal(model)


if __name__ == "__main__":
    shareVar = multiprocessing.Manager().dict()
    t_1 = multiprocessing.Process(target=Track().main, args=(shareVar, bluecode,))
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
