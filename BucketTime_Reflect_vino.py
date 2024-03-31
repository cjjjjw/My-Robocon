"""
brief :Editor cjh666
"""

import argparse
import numpy as np
import time
from collections import deque
import multiprocessing
from Map_Reflect_utils import if_model, Map_Reflect
from utils_track import vino as ov
import cv2

# 定义棋子
bluecode = 1.0
redcode = -1.0


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


def FUCKING(queue, share_var, shape_x, code):
    # 先进行按列排序
    color_code = code
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
                for i in range(len(data_coord)):
                    for x in range(len(x_coordinate)):
                        # print(data_coord,mediator_dict["height"])
                        if abs(data_coord[i][0] - x_coordinate[x]) < mediator_dict["width"] / 2 and abs(
                                data_coord[i][1]) > \
                                mediator_dict["height"] - (mediator_dict["width"] / 2) / 1.2:
                            cluster[x].append(data_coord[i][1:])
                            del cluster[x][0]
                cluster = np.array(cluster)
                x_coordinate = abs(np.array(sorted(mediator_dict["col"])) - shape_x / 2)
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
    while True:
        Best_ROI_now = share_var["Best_ROI"]
        track_target_now = share_var["Best_target"]
        track_flag = share_var["Track_flag"]
        if len(stack) != 0:
            Frame = stack.pop()
            if track_flag:
                update_counter += 1
                if track_target_now != track_target_last and update_counter < 50 and track_target_now:
                    track_target_last = track_target_now
                    tracker = cv2.TrackerKCF.create()
                    tracker.init(Frame, Best_ROI_now)
                    update_counter = 0
                    # 避免同时更新和初始化,设置了一个更新时长控制
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
        try:
            cap = cv2.VideoCapture(int(source))
        except Exception:
            pass
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

    def run_bucket(self, model):
        while True:
            if len(Cap_data) != 0:
                frame = Cap_data.pop()
                start = time.time()
                self.mediator_dict["ball_deque_coord"], self.mediator_dict["col"], self.mediator_dict[
                    "width"], self.mediator_dict["low"] = deque(), deque(), 0, 0
                frame, det = model.run(frame)
                # copy to become another image
                if len(det):
                    for xyxy, conf, cls in reversed(det):  # 识别物的类别像素坐标，置信度，物体类别号
                        mid_pos = [int((int(xyxy[0]) + int(xyxy[2])) / 2),
                                   int((int(xyxy[1]) + int(xyxy[3])) / 2)]
                        cv2.circle(frame, mid_pos, 1, (0,255,0), 3)  # 圆心
                        cv2.putText(frame, f"{mid_pos}", mid_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        # 在Bucket中,0 is bucket,1 is red ,2 is blue
                        if cls == 1:
                            self.mediator_dict["ball_deque_coord"].append([*mid_pos, int(redcode)])
                        elif cls == 2:
                            self.mediator_dict["ball_deque_coord"].append([*mid_pos, int(bluecode)])
                        elif cls == 0:
                            self.mediator_dict["width"] = int((xyxy[2] - xyxy[0]))
                            self.mediator_dict["height"] = int(xyxy[1])
                            self.mediator_dict["low"] = int(xyxy[3])
                            self.mediator_dict["col"].append(mid_pos[0])
                    queue.put(self.mediator_dict)
                    KCF_data["Track_flag"] = True
                else:
                    KCF_data["Track_flag"] = False
                # 数据交换
                self.share_data["Reflect_data"] = Reflect_data
                KCF_data["Best_ROI"] = Reflect_data["Best_ROI"]
                KCF_data["Best_target"] = Reflect_data["Best"]
                # 以下是向外输送
                if KCF_data["Track_data"]:
                    self.share_data["Best_x"] = KCF_data["Track_data"][0][0]
                    cv2.rectangle(frame, KCF_data["Track_data"][1], KCF_data["Track_data"][2], (255, 0, 0), 2, 1)
                end = time.time()
                fps = 1 / (end - start)
                cv2.putText(frame, f"{round(fps, 1)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                self.share_data["im0"] = frame

    # This function is used to choose argument,which is called init function
    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_path', type=str, default="/home/nuc2/PycharmProjects/yolov5-master/best_bucket.xml",
                            help='model path')
        parser.add_argument('--weights_path', nargs='+', type=str, default="/home/nuc2/PycharmProjects/yolov5-master"
                                                                           "/best_bucket.bin",
                            help='weights path or triton URL')
        parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--iou-thres', type=float, default=0.55, help='NMS IoU threshold')
        parser.add_argument('--device', default='GPU', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--name-porcess', default='BucketTime', help='name to processs--cjh')
        parser.add_argument('--source', type=str, default="/home/nuc2/PycharmProjects/yolov5-master/source_video/6.mp4",
                            help='')
        opt = parser.parse_args()

        return opt

    # start function
    def main(self, shareVar, code):
        """
            shareVar:多进程间的共享变量
            code:确定是红球还是蓝球
        """
        global queue, Reflect_data, Cap_data, KCF_data
        queue = multiprocessing.Queue()
        Reflect_data = multiprocessing.Manager().dict()
        KCF_data = multiprocessing.Manager().dict()
        # 创建两条视频流
        Cap_data = multiprocessing.Manager().list()
        Cap_data2 = multiprocessing.Manager().list()
        # 初始化赋值["Best_ROI"],Reflect_data["Best_ROI"]避免先启动读不到数据
        KCF_data["Best_ROI"], Reflect_data["Best_ROI"], KCF_data["Track_data"], Reflect_data["Best"], KCF_data[
            "Best_target"], KCF_data["Track_flag"] = (), (), [], None, None, False
        # 为了将数据送出，构建一个共享字典
        self.share_data = shareVar
        # 使其为全局变量，不然无法添加
        opt = self.parse_opt()
        # 启动进程
        p1 = multiprocessing.Process(target=cap_capture, args=(Cap_data, Cap_data2, vars(opt)["source"], 30, 30))
        p2 = multiprocessing.Process(target=FUCKING, args=(queue, Reflect_data, 640, code))
        p3 = multiprocessing.Process(target=KFC_Thursday, args=(Cap_data2, KCF_data))
        p1.start()
        p2.start()
        p3.start()
        # 便于关闭所有进程
        self.share_data["pid"] = [p1.pid, p2.pid, p3.pid]
        del vars(opt)["source"]
        vino_Bucket = ov.Vino(**vars(opt))
        self.run_bucket(vino_Bucket)


if __name__ == "__main__":
    shareVar = multiprocessing.Manager().dict()
    t_1 = multiprocessing.Process(target=Track().main, args=(shareVar, bluecode))
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
