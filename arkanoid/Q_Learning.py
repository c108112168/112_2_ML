import random
import numpy as np
import csv

class QLearning:
    def __init__(self, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.last_ball_position = None
        self.q_table = {}
        self.epsilon = epsilon  # 探索率
        self.alpha = alpha      # 學習率
        self.gamma = gamma      # 折扣因子
        self.last_distance_x = None  # 初始化 last_distance_x
        self.ball_speed_x = 0
        self.ball_speed_y = 0

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
                if len(row) == 4:  # 檢查行是否包含正確的數據
                    try:
                        state = int(row[0])
                        action_values = [float(row[1]), float(row[2]), float(row[3])]
                        self.q_table[state] = action_values
                    except ValueError:
                        print("Invalid row:", row)  # 輸出無效的行
                else:
                    print("Invalid row:", row)  # 輸出無效的行

    def state(self, scene_info):
        platform_x = scene_info['platform_1P'][0]  # 假設使用玩家1的板子
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

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]  # [左移, 不動, 右移]

        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(self.q_table[state])

        return action

    def learn(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0]

        q_predict = self.q_table[state][action]
        q_target = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (q_target - q_predict)

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.ball_served = False
        self.side = ai_name
        self.ql = QLearning()  # 創建 QLearning 對象
        self.previous_state = None
        self.previous_action = None

    def update(self, scene_info, keyboard=[], *args, **kwargs):
        """
        根據接收到的場景信息生成命令
        """
        if scene_info["status"] != "GAME_ALIVE":
            # 重置遊戲狀態
            self.previous_state = None
            self.previous_action = None
            return "RESET"
            
        # 從 CSV 文件讀取 Q 表
        self.ql.q_table_read("q_table.csv")
        
        current_state = self.ql.state(scene_info)

        if not self.ball_served:
            self.ball_served = True
            return "SERVE_TO_LEFT"
        else:
            # 根據當前狀態選擇動作
            action_idx = self.ql.choose_action(current_state)
            actions = ["MOVE_LEFT", "NONE", "MOVE_RIGHT"]
            action = actions[action_idx]
            
            # 獎勵系統 (這裡可以根據具體情況修改)
            reward = self.ql.reward(current_state)

            # 在新狀態下學習
            if self.previous_state is not None and self.previous_action is not None:
                self.ql.learn(self.previous_state, self.previous_action, reward, current_state)
                
            self.ql.q_table_save("q_table.csv")

            # 更新前一狀態和動作
            self.previous_state = current_state
            self.previous_action = action_idx
            return action

    def reset(self):
        """
        重置狀態
        """
        self.ball_served = False