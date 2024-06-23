import numpy as np
import pandas as pd
import random
import csv

class QLearning:
    def __init__(self, eGreedy=0.1, learning_rate=0.7, discount_factor=0.9):
        self.eGreedy = eGreedy
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}
        self.last_distance_x = None  # 初始化 last_distance_x
        self.ball_speed_x = 0
        self.ball_speed_y = 0
        self.last_ball_position = None


    def choose_action(self, state):
#        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]  # [左移, 不動, 右移]

        if random.uniform(0, 1) < self.eGreedy:
            action = random.randint(0, 2)
        else:
            action = np.argmax(self.q_table[state])

        return action

    def q_table_save(self, file_path):
        """
        將 Q 表儲存為 CSV 文件
        """
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for state, action_values in self.q_table.items():
                writer.writerow([state, action_values[0], action_values[1], action_values[2]])
    
    def q_table_read(self, file_path):
        """
        從 CSV 文件讀取 Q 表
        """
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) == 6:  # 檢查行是否包含正確的數據
                    try:
                        state = (int(row[0]))
                        action_values = [float(row[1]), float(row[1]), float(row[3])]
                        self.q_table[state] = action_values
                    except ValueError:
                        print("Invalid row_1:", row)  # 輸出無效的行
                else:
                    print("Invalid row_2:", row)  # 輸出無效的行

    def learn(self, state, action, reward, next_state):
#        state = tuple(state)
#        next_state = tuple(next_state)
        if state not in self.q_table:
            self.q_table[state] = [0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0]

        predict = self.q_table[state][action]
        target = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (target - predict)

    def state(self, scene_info):
        platform_x = scene_info['platform'][0]  # 假設使用玩家1的板子
        ball_x, ball_y = scene_info['ball']

        if self.last_ball_position is None:
            self.last_ball_position = (ball_x, ball_y)
        elif scene_info['frame'] % 5 == 0:
            # 計算座標差來獲取球的速度
            self.ball_speed_x = ball_x - self.last_ball_position[0]
            self.ball_speed_y = ball_y - self.last_ball_position[1]
            self.last_ball_position = (ball_x, ball_y)

        # 計算球距離板子中心點的距離（有方向性）
        distance_x = ball_x - platform_x

        return distance_x
        
    def reward(self, current_state):
        distance_x = current_state

        if self.last_distance_x is None:
            self.last_distance_x = distance_x
            return 0

        # 獎勵計算：如果球變得更靠近板子中心，則加分
        if abs(distance_x) > abs(self.last_distance_x):
            reward = 50
        else:
            reward = -10

        self.last_distance_x = distance_x
        return reward

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.ai_name = ai_name
        self.q_learning = QLearning()
        self.ball_served = False
        self.state = None
        self.action = None

    def update(self, scene_info, *args, **kwargs):
        if scene_info["status"] in ["GAME_OVER", "GAME_PASS"]:
            return "RESET"
#        self.q_learning.q_table_read('q_table.csv')

        if not self.ball_served:
            command = "SERVE_TO_LEFT"
            self.ball_served = True
            self.state = self.q_learning.state(scene_info)
            self.action = 1  # 初始不動作
        else:
            state_ = self.q_learning.state(scene_info)
            reward = self.q_learning.reward(self.state, state_)
            self.q_learning.learn(self.state, self.action, reward, next_state=state_)
            self.q_learning.q_table_save('q_table.csv')
            self.state = state_

            self.action = self.q_learning.choose_action(self.state)
            if self.action == 0:
                command = "MOVE_LEFT"
            elif self.action == 1:
                command = "STAY"
            else:
                command = "MOVE_RIGHT"

        return command

    def reset(self):
        self.ball_served = False
        self.state = None
        self.action = None