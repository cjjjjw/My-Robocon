import time
import if_model, Map_Reflect
import numpy as np


class Game:
    def __init__(self):
        self.env = Map_Reflect.State()
        self.env.map_init()  # 地图启动
        self.model = if_model.Judgment
        self.state = np.zeros((3, 5))
        self.x_coordinate = np.array([3, 2, 0, 4, 5])
        self.done = False

    def set_state(self):  # 自定义开始地图
        print(self.env.observation)
        User_input_nums = int(input("nums:"))
        if User_input_nums > 0:
            for _ in range(User_input_nums):
                color = int(input('color(blue is 1,red is -1):'))
                col = int(input("which one:"))
                self.env.step(col, color)

    def User_choose(self):
        # User_input_nums = int(input("nums:"))
        # if User_input_nums > 0:
        #     for _ in range(User_input_nums):
        #         # color = int(input('color(blue is 1,red is -1):'))
        #         col = int(input("which one:")) - 1
        #         next_state, _, self.done = self.env.step(col, -1)
        #         self.state = next_state
        for _ in range(np.random.randint(1, 2)):
            action,min_x = self.model(self.x_coordinate, self.state, -1).main_decision()
            print(action,min_x, "com-actions")
            next_state, _, self.done = self.env.step(action,-1)
            self.state = next_state

    def Computer_choose(self):  # 给机器人随机放两个的机会
        for _ in range(np.random.randint(1, 2)):
            action,min_x = self.model(self.x_coordinate, self.state, 1).main_decision()
            print(action,min_x, "com-actions")
            next_state, _, self.done = self.env.step(action, 1)
            self.state = next_state

    def play(self):
        self.state = self.env.reset()  # 游戏画面初始化
        # time.sleep(1)
        # 随机谁先起手
        color_s = np.random.randint(3, 5)
        while True:
            color_s += 1
            if color_s % 2 == 0:
                self.User_choose()
            else:
                self.Computer_choose()
            time.sleep(0.1)
            if self.done:
                self.done = False
                self.state = self.env.reset()


if __name__ == "__main__":
    # 获取环境的状态
    # 实例化地图对象
    User = Game()
    User.play()
