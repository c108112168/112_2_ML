import random
import numpy as np
import csv
import math


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
    '''    
    def print_reward_info(self, state, action, reward):
        action_str = ["MOVE_LEFT", "NONE", "MOVE_RIGHT"][action]
        print(f"state_= {self.previous_state}, state = {state}, action = {action_str}, reward = {reward}")
        print("Invalid row:")  # 輸出無效的行
    '''    
    def q_table_save(self, file_path):
        """
        將 Q 表儲存為 CSV 文件
        """
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for state, action_values in self.q_table.items():
                writer.writerow([state, action_values[0], action_values[1]])

    def q_table_read(self, file_path):
        """
        從 CSV 文件讀取 Q 表
        """
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) == 3:  # 檢查行是否包含正確的數據
                    try:
                        state = int(row[0])
                        action_values = [float(row[1]), float(row[2])]
                        self.q_table[state] = action_values
                    except ValueError:
                        print("Invalid row:", row)  # 輸出無效的行
                else:
                    print("Invalid row:", row)  # 輸出無效的行

    def state(self, scene_info):
        # 取得板子和球的當前位置
        platform_1P_x = scene_info['platform_1P'][0]  # 玩家1的板子位置（X座標）
        platform_2P_x = scene_info['platform_2P'][0]  # 玩家2的板子位置（X座標）
        ball_x, ball_y = scene_info['ball']           # 球的當前位置
        ball_speed_x, ball_speed_y = scene_info['ball_speed']  # 球的當前速度
        o_ball_speed_x, o_ball_speed_y = scene_info['ball_speed']  # 球的當前速度

        # 設定桌子的寬度和兩個板子的高度
        table_width = 189
        table_height = 500
        platform_1P_y = 420  # 玩家1板子的高度
        platform_2P_y = 80   # 玩家2板子的高度
        
        original_ball_x = ball_x  # 保存原始的球的位置
        original_ball_y = ball_y  # 保存原始的球的位置
        time_to_wall = 0
        # 避免球的速度為零導致的除零錯誤
        if ball_speed_y == 0:
            ball_speed_y = 1e-5  # 設置一個很小的值來代替零

        while ball_y<platform_1P_y and ball_y>platform_2P_y:
            if ball_speed_y > 0:  # 球向下移動
                time_to_platform_1P = (platform_1P_y - ball_y) / ball_speed_y
                future_ball_x = ball_x + ball_speed_x * time_to_platform_1P

                if future_ball_x < 0 or future_ball_x > table_width:
                    if ball_speed_x > 0:
                        time_to_wall =  math.ceil((table_width - ball_x) / ball_speed_x)
                        ball_x += ball_speed_x * int(time_to_wall)
                    else:
                        if (ball_x % ball_speed_x) != 0:
                            ball_x += abs(ball_x % ball_speed_x)
                        time_to_wall = -ball_x / ball_speed_x
                        
                        ball_x += ball_speed_x * int(time_to_wall) 

                    ball_y += ball_speed_y * int(time_to_wall)
                    ball_speed_x = -ball_speed_x  # 反射
                else:
                    ball_x = future_ball_x
                    ball_y = platform_1P_y
                    break

            else:  # 球向上移動
                time_to_platform_2P = (platform_2P_y - ball_y) / ball_speed_y
                future_ball_x = ball_x + ball_speed_x * time_to_platform_2P

                if future_ball_x < 0 or future_ball_x > table_width:
                    if ball_speed_x > 0:
                        time_to_wall =  math.ceil((table_width - ball_x) / ball_speed_x)
                        ball_x += ball_speed_x * int(time_to_wall)
                    else:
                        if (ball_x % ball_speed_x) != 0:
                            ball_x += abs(ball_x % ball_speed_x)
                        time_to_wall = -ball_x / ball_speed_x
                        ball_x += ball_speed_x * int(time_to_wall)

                    
                    ball_y += ball_speed_y * int(time_to_wall)
                    ball_speed_x = -ball_speed_x  # 反射
                else:
                    ball_x = future_ball_x
                    ball_y = platform_2P_y
                    break

        # 計算球距離板子中心點的距離（有方向性）
        if ball_speed_y > 0:
            distance_x = ball_x - (platform_1P_x + 20)
        else:
            distance_x = ball_x - (platform_2P_x + 20)
        print(f"distance_x: {distance_x}")
# 打印出球的座標，速度和預測落點
        #print(f"Ball position: ({original_ball_x}, {original_ball_y}), Ball speed: ({o_ball_speed_x}, {o_ball_speed_y}), Predicted landing position: {ball_x}, time_to_wall: {time_to_wall}")

        return int(distance_x)

    def reward(self, current_state):
        distance_x = current_state

        if self.last_distance_x is None:
            self.last_distance_x = distance_x
            return 0

        # 獎勵計算：如果球變得更靠近板子中心，則加分
        if abs(distance_x) > abs(self.last_distance_x):
            reward = -5*(abs(distance_x) - abs(self.last_distance_x))
        else:
            reward = -8*(abs(distance_x) - abs(self.last_distance_x))

        self.last_distance_x = distance_x
        

        return reward

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0]  # [左移, 不動, 右移]

        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 1)
        else:
            max_value = max(self.q_table[state])
            max_indices = [i for i, v in enumerate(self.q_table[state]) if v == max_value]
            action = random.choice(max_indices)


        return action

    def learn(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0]

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
            # 從 CSV 文件讀取 Q 
            self.ql.q_table_read("q_table.csv")
            return "RESET"
            
        
        
        current_state = self.ql.state(scene_info)

        if not self.ball_served:
            self.ball_served = True
            return "SERVE_TO_LEFT"
        else:
            # 根據當前狀態選擇動作
            action_idx = self.ql.choose_action(current_state)
            actions = ["MOVE_LEFT", "MOVE_RIGHT"]
            #if current_state < 0:
            #    action_idx = 0
            #else:
            #    action_idx = 1
            action = actions[action_idx]
            
            # 獎勵系統 (這裡可以根據具體情況修改)
            reward = self.ql.reward(current_state)
            #self.ql.print_reward_info(current_state, action_idx, reward)

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
