"""
brief :由于pyplot与tk各有局限性在多线程中的使用体验差，此处使用cv2构建地图
"""
import random

import cv2
import time
import numpy as np
import threading

# 定义棋子
bluecode = 1.0
redcode = -1.0
backcode = 0.0
map_size = np.zeros((3, 5))


class Map(object):
    def __init__(self):
        self.weight = 800
        self.height = 600
        self.title = "white board"
        self.map = None
        self.done_list = np.zeros((1, 5))
        self.counter = [0, 0, 0, 0.0]  # 计算成功，失败，场次，和成功率

    def map_init(self):
        # 构造白色地图
        self.map = np.ones((self.height, self.weight, 3), dtype=np.uint8) * 255
        cv2.rectangle(self.map, (20, 200), (140, 600), (0, 0, 0), 3, -1)
        cv2.rectangle(self.map, (180, 200), (300, 600), (0, 0, 0), 3, -1)
        cv2.rectangle(self.map, (340, 200), (460, 600), (0, 0, 0), 3, -1)
        cv2.rectangle(self.map, (500, 200), (620, 600), (0, 0, 0), 3, -1)
        cv2.rectangle(self.map, (660, 200), (780, 600), (0, 0, 0), 3, -1)
        t1 = threading.Thread(target=self.show, args=(), daemon=True)
        # 设置守护线程,即主线程启动才会启动子线程
        t1.start()

    # 解释状态与处于地图的哪一步
    def action(self, row, col, state_n):
        insert_map = [[(80, 266), (240, 266), (400, 266), (560, 266), (720, 266)],
                      [(80, 399), (240, 399), (400, 399), (560, 399), (720, 399)],
                      [(80, 532), (240, 532), (400, 532), (560, 532), (720, 532)]]
        if state_n == bluecode:
            cv2.circle(self.map, insert_map[row][col], 60, (255, 0, 0), -1)
        elif state_n == redcode:
            cv2.circle(self.map, insert_map[row][col], 60, (0, 0, 255), -1)

    # 展现画布，多线程共行，且尽量避免线程锁，共享变量只有一处能改变
    def show(self):
        while True:
            cv2.imshow(self.title, self.map)
            cv2.waitKey(1)


class State(Map):
    def __init__(self):
        super(State, self).__init__()
        self.observation = np.zeros((3, 5))  # reset state,复调用前面的地图

    def add_action(self, action, color, observation):
        for i in range(1, len(observation[0]) + 1):
            if observation[action][-i] == 0:
                observation[action][-i] = color
                self.action(-i, action, color)
                break

    def take_actions(self, action, color):
        # 将observation转置，方便判断是否有球
        observation = self.observation.T
        if 0 not in self.observation:
            return False
        # 各个框的选择
        self.add_action(action, color, observation)
        # 将更新矩阵
        self.observation = observation.T
        return True

    # 判断是否胜利
    def is_done(self, action):

        done_s, reward_s = self.type_1(action)
        if done_s == 1:
            cv2.putText(self.map, "Win", (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.counter[2] += 1
            self.counter[0] += 1
        elif done_s == -1:
            cv2.putText(self.map, "Lose", (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.counter[2] += 1
            self.counter[1] += 1
        return done_s, reward_s

    # 有两种情况胜利，失败
    # 大胜的条件
    def type_1(self, action):
        insert_map = [(40, 186), (200, 186), (360, 186), (520, 186), (680, 186)]
        observation = self.observation.T
        # 以下两步不为必要
        if np.sum(self.done_list == -1) > 2:  # 通过判断1与-1确定胜利与否，这样即可通过改变red和blue的赋值，改变胜利的条件
            self.done_list = np.zeros((1, 5))
            return -1, -1000  # 1和-1都是完成
        elif np.sum(self.done_list == 1) > 2:
            self.done_list = np.zeros((1, 5))
            return 1, 1000

        if 0 in observation:  # 判断有没有放满
            # 先判断顶端有没有放满
            # 判断哪一个在顶端,顶端有颜色，那肯定是那一筐已经放满了

            if observation[action][0] == -1:
                if np.sum(observation[action]) < 0:  # 判断球的类型
                    self.done_list[0][action] = -1  # 赋值于条件矩阵
                    cv2.putText(self.map, "TOP-Red", insert_map[action], cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255),
                                2)  # 可视化
                return 0, 0  # 游戏继续
            elif observation[action][0] == 1:  # 判断球的类型
                if np.sum(observation[action]) > 0:
                    self.done_list[0][action] = 1
                    cv2.putText(self.map, "TOP-Blue", insert_map[action], cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (255, 0, 0),
                                2)
                return 0, 30  # 判断是否堆满则奖励
            else:
                if observation[action][1] == 1:  # 第二层是正确色的，非常抑制
                    return 0, 30
                elif observation[action][2] == 1 and observation[action][1] == 0:  # 第1层是正确色的，非常赞成
                    return 0, 30  # 比上第二层优先
                else:
                    return 0, 0  # 第1或第2层不是是正确色的，不影响游戏，不计分

        else:  # 没有大胜开始判断谁放的多
            self.done_list = np.zeros((1, 5))  # 将检查数组清空
            return self.type_2()

    # 有两种情况胜利，失败
    # 球数的条件
    def type_2(self):
        if np.sum(self.observation == 1) > np.sum(self.observation == -1):
            return 1, 0  # 胜利最大奖励
        else:
            return -1, 0  # 有些时候失败无法避免，因此避免大败即可

    # 清除画布,复原数组
    def reset(self):
        time.sleep(0.05)
        self.map = np.ones((self.height, self.weight, 3), dtype=np.uint8) * 255
        cv2.rectangle(self.map, (20, 200), (140, 600), (0, 0, 0), 3, -1)
        cv2.rectangle(self.map, (180, 200), (300, 600), (0, 0, 0), 3, -1)
        cv2.rectangle(self.map, (340, 200), (460, 600), (0, 0, 0), 3, -1)
        cv2.rectangle(self.map, (500, 200), (620, 600), (0, 0, 0), 3, -1)
        cv2.rectangle(self.map, (660, 200), (780, 600), (0, 0, 0), 3, -1)
        self.observation = np.zeros((3, 5))
        return self.observation

    # 这是地图的输出函数,next_state, reward, done
    def step(self, action, color):
        if self.counter[2] >= 30:
            self.counter = [0, 0, 0, 0]  # 大于30重新计算
        self.take_actions(action, color)
        done_s = False
        done_ready, reward_s = self.is_done(action)
        # print(reward_s)
        if done_ready != 0:
            done_s = True  # 表明状态已经完成
        return self.observation, reward_s, done_s
        # 状态为


if __name__ == "__main__":
    """
        :param:以下为功能展示
    """
    state = State()  # map被state继承了
    # 地图初始化
    state.map_init()
    i = 0
    counter = 0
    for _ in range(10):
        print(counter)
        time.sleep(0.1)
        state.reset()
        while True:
            next_state, reward, done = state.step(i, random.randint(-1, 1))
            print(next_state, reward, done, "状态")
            print(next_state.T)
            counter += reward
            if done:
                break
            i += 1
            if i > 4:  # 大于5清0
                i = 0
            time.sleep(1000)
    # 清除画布
    time.sleep(50)
