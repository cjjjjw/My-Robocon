import numpy as np
import cv2
import time
from Map_Reflect_utils import Realsense
from collections import deque
import multiprocessing
from Map_Reflect_utils import if_model, Map_Reflect
import serial
import logging

# 定义棋子
bluecode = 1.0
redcode = -1.0
backcode = 0.0
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("Depth")


# 数组的裁切
def delete(arr):
    SHAPE = arr.shape
    DELETE = 0
    arr[:, DELETE] = arr[:, SHAPE[1] - 1]
    arr2 = arr[:, :SHAPE[1] - 1].T
    arr2 = np.squeeze(arr2).tolist()
    return arr2


def ser_open(port0="/dev/ttyUSB2"):
    ser2 = serial.Serial()
    ser2.baudrate = 115200  # 波特率
    ser2.port = port0
    if ser2.isOpen():
        print("串口打开成功")
    else:
        ser2.open()
    return ser2.isOpen(), ser2


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
        ret, ser1 = ser_open("/dev/ttyUSB0")
        while ret:
            ser1.reset_input_buffer()  # 清除输入缓冲区
            # print(queue2.get())
            ser1.write(queue2.get())
    except:
        logger.info(f"Sender ser is Died")


def FUCKING(share_var, code, queue1, queue2):
    # 先进行按列排序
    env = Map_Reflect.State()
    env.map_init(False)  # 地图启动
    model = if_model.Judgment
    share_var['Reflect_img'] = env.map
    x_coordinate = np.array([5, 3, 0, 2, 4])
    np.random.shuffle(x_coordinate)
    # 打开管道接收数据
    last_ser = 0
    while True:
        dist_detect = share_var["depth"]
        action_best, min_x_best = model(x_coordinate, env.observation, code).main_decision()
        next_state, _, _ = env.step(action_best, 2)
        try:
            # 如此放入最先考虑的是中间，因此直接赋值中间是最小的
            action_better, min_x_better = model(x_coordinate, next_state, code).main_decision()
            next_state, _, _ = env.step(action_better, 3)
            # 状态的表示
            share_var['Best'] = action_best
            share_var['Better'] = action_better
        except Exception:
            print("No other situation")
        # 做完预测之后，执行状态恢复避免反复判断
        env.reflect(next_state)
        share_var['Reflect_img'] = env.map
        env.map_reset()
        # 先传输移动命令到，下位机ser.send(best_action)
        # print(str(f"5A{action_best}{0}"))
        queue2.put(str(f"5A{action_best}{0}00000000000000").encode("gbk"))
        set_flag = queue1.get()
        # 通过接收下位机数据来确定是否达到位置和当前的位置，否则不改变当前状态，而深度为了确定这个位置是否被对面放球了
        # 一直更新最佳位置，但是却不是释放放球的命令
        # 当发生反转信号，set_flag==1，last_ser==0，则发生一次状态记录
        print(set_flag, " now_sent")
        print(last_ser, "last_ser")
        if set_flag == 1 and last_ser != 1:
            print("已经到达眶前")
            if not dist_detect:  # 有三个
                # 缺多少补多少
                for i in env.observation.T[action_best][::-1]:
                    if i == 0:
                        env.observation, _, _ = env.step(action_best, -int(code))
                        print("不放入切换另一个最佳位置")
            elif dist_detect < 0.35:  # 有两个
                # 缺下面两层，补下面多少
                if 0 in env.observation.T[action_best][1:]:
                    for i in env.observation.T[action_best][:0:-1]:
                        if i == 0:
                            env.observation, _, _ = env.step(action_best, -int(code))
                            print("不放入切换另一个最佳位置")
                else:
                    env.observation, _, _ = env.step(action_best, int(code))
                    queue2.put(str(f"5A{action_best}{1}00000000000000").encode("gbk"))
                    print(f"放入{action_best}")
            elif dist_detect < 0.45:  # 有一个
                # 缺下面一层，确认是否补下面一层
                if 0 in env.observation.T[action_best][2:]:
                    env.observation, _, _ = env.step(action_best, -int(code))
                    print("不放入切换另一个最佳位置")
                else:
                    env.observation, _, _ = env.step(action_best, int(code))
                    queue2.put(str(f"5A{action_best}{1}00000000000000").encode("gbk"))
                    print(f"放入{action_best}")
            elif dist_detect < 0.55:  # 空的
                env.observation, _, _ = env.step(action_best, int(code))
                queue2.put(str(f"5A{action_best}{1}00000000000000").encode("gbk"))
                print(f"放入{action_best}")
        # 发生反转信号前，都与set_flag 相同
        last_ser = set_flag


class Track(object):
    # 继承plot的类,传入一个共享变量在并行处理的时候可以读取到BucketTime的各种信息
    def __init__(self):
        self.realsense_cap = None
        self.share_data = dict
        self.Reflect_data = dict
        self.randnum = 20
        # 创建深度检测点的范围
        self.pos = np.arange(-20, 21, 5)

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

    def run_normal(self):
        self.realsense_cap = Realsense.realsense(0)
        cap = self.realsense_cap.cam_init(640)
        while True:
            # 需要reset一下mediator_dict
            start = time.time()
            distance_list = deque()
            # color_image, _, depth_intrin, aligned_depth_frame = self.realsense_cap.cam_run(self.cap)
            color_image, depth_colormap, depth_intrin, aligned_depth_frame = self.realsense_cap.cam_run(cap)
            for x in range(len(self.pos)):
                mid_pos_up = (int((color_image.shape[1]) / 2 - self.pos[x]),
                              int((color_image.shape[0]) / 2 - self.pos[x]))  # 确定索引深度的中心像素位置左上角和右下角相加在/2
                cv2.circle(color_image, mid_pos_up, 2, (0, 255, 0), 2)
                z1, y1, x1 = self.realsense_cap.depth_to_data(depth_intrin, aligned_depth_frame, mid_pos_up)
                if z1:
                    distance_list.append(round(z1, 3))
            distance_list = np.array(distance_list)
            distance_list = np.sort(distance_list)[
                            self.randnum // 2 - self.randnum // 4:self.randnum // 2 + self.randnum // 4]
            mean_distance = np.mean(distance_list)
            if not mean_distance > 0:
                mean_distance = 0
            Reflect_data["depth"] = float(mean_distance)
            # 设置清空画图的情况，且从UI部接收这个数据
            # images_real = np.hstack((color_image, depth_colormap))
            # cv2.putText(images_real, f"{round(float(mean_distance), 3)}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1,
            #             (255, 0, 0), 2)
            cv2.putText(color_image, f"{round(float(mean_distance), 3)}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (255, 0, 0), 2)
            end = time.time()
            try:
                fps = 1 / (end - start)
                cv2.putText(color_image, f"{round(fps, 1)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                # cv2.putText(images_real, f"{round(fps, 1)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            except:
                pass
            self.share_data["Reflect_data"] = Reflect_data
            self.share_data["im0"] = color_image
            self.share_data["depth"] = float(mean_distance)

    # start function,外部提供串口ser
    def main(self, shareVar, code):
        global Reflect_data
        self.share_data = shareVar
        # queue = multiprocessing.Queue()
        # 创建两条视频流
        Reflect_data = multiprocessing.Manager().dict()
        Reflect_data["depth"], Reflect_data["Best"], Reflect_data[
            "Better"] = float(0), int(2), int(2)
        Receiver_queue = multiprocessing.Manager().Queue(maxsize=2)
        Sender_queue = multiprocessing.Manager().Queue(maxsize=2)
        p1 = multiprocessing.Process(target=FUCKING, args=(Reflect_data, code, Receiver_queue, Sender_queue))
        p2 = multiprocessing.Process(target=Receiver, args=(Receiver_queue,))
        p3 = multiprocessing.Process(target=Sender, args=(Sender_queue,))
        p2.start()
        p3.start()
        p1.start()
        self.share_data["pid"] = [p1.pid, p2.pid, p3.pid]
        self.run_normal()


if __name__ == "__main__":
    shareVar = multiprocessing.Manager().dict()
    # shareVar["code"] = redcode
    track = Track()
    p3 = multiprocessing.Process(target=track.main, args=(shareVar, redcode,))
    p3.start()
    while True:
        try:
            cv2.imshow("1231", shareVar['im0'])
            cv2.imshow('321', shareVar["Reflect_data"]["Reflect_img"])
            print('最佳选择', shareVar["Reflect_data"]["Best"])
            print('次要选择', shareVar["Reflect_data"]["Better"])
            print('深度', shareVar["depth"])
            cv2.waitKey(1)
        except Exception:
            pass
    # track.run_normal()
