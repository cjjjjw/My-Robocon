import numpy as np
from collections import deque


class Judgment:
    def __init__(self, x_coordinate: np.ndarray, now_state: np.ndarray, color_code, power=False):
        self.state = now_state
        # 做一个x坐标的居中处理，去靠近最近的坐标
        self.x_coordinate = x_coordinate
        self.color = color_code
        self.force = [0, 0, 0, 0, 0]
        self.power = power

    # 第一层的抉择
    def main_decision(self):
        """
        state.T:
        [[0. 0. 1.]
         [0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]
         [0. 0. 0.]]
        :return:
        """
        if 0 in self.state[1]:
            # print("step force")
            for index in range(len(self.state.T)):
                # 优先判断拦截第三层
                if self.state.T[index][1] != 0 and self.state.T[index][0] == 0:
                    pass
                else:
                    self.force[index] = self.color
            if 0 in self.force:
                index, min_x = self.min_choice(np.array(self.force))
                return int(index[0]), int(min_x)

        if self.power:  # 当我方实力更强，则使用策略一快速将球填满
            if 0 in self.state[-1] or 0 in self.state[-2]:
                # 首先拦截第一层是自己方的球
                # 其次拦截第一层
                # 第一层满了，在开始拦截第二层
                return self.step2_best()
            else:
                # 当第二层没有了，开始拦截第三层
                return self.step3(np.array(self.state[-3]))

        elif not self.power:
            if 0 in self.state[-1]:
                # 首先拦截第一层
                return self.step1(np.array(self.state[-1]))
            elif 0 in self.state[-2]:
                return self.step2_best()
            else:
                # 当第二层没有了，开始拦截第三层
                return self.step3(np.array(self.state[-3]))

    def step1(self, state_row):
        # 第一层做一个最近点的放入
        # print('step1')
        index, min_x = self.min_choice(state_row)
        return int(index[0]), int(min_x)

    def step2_best(self):
        # print('step2')
        # 直接做一个状态检测
        min_x_ = np.Inf
        target_best_index = None
        for i in range(len(self.state[0])):
            if self.state[1][i] == 0 and self.state[2][i] == self.color:
                if self.x_coordinate[i] < min_x_:
                    min_x_ = self.x_coordinate[i]
                    # 最好的情况是，二层下方是自己的球
                    target_best_index = i
        if target_best_index is not None:
            return target_best_index, min_x_
        else:
            if 0 in self.state[-1]:
                return self.step1(np.array(self.state[-1]))
            else:
                return self.step2_better()

    def step2_better(self):
        min_x = np.Inf
        target_better_index = None
        for i in range(len(self.state[0])):
            if self.state[1][i] == 0:
                if self.x_coordinate[i] < min_x:
                    min_x = self.x_coordinate[i]
                    # 次好的情况是，二层完成放置
                    target_better_index = i
        return target_better_index, min_x

    def step3(self, state_row):
        # print('step3')
        index, min_x = self.min_choice(state_row)
        return int(index[0]), int(min_x)

    def min_choice(self, state_row):
        # 如果是有球的直接归0，没有球的
        # 返回一个最近距离，且是空框的索引坐标
        # print(state_row)
        # print(np.where(state_row == 0))
        # print(self.x_coordinate[np.where(state_row == 0)])
        min_x_coord = np.min(self.x_coordinate[np.where(state_row == 0)])
        min_index = np.where(self.x_coordinate == min_x_coord)
        return min_index, min_x_coord


if __name__ == "__main__":
    state = np.array([[0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0.],
                      [-1., 0., 0., 0., 0.]])
    judge = Judgment(np.array([12, 34, 23, 45, 66]), state, 1)
    print(judge.main_decision())
