import random
import numpy as np
import csv
import math


import pygame
class QLearning:
    def __init__(self, learning_rate=0.1, reward_decay=0.9, e_greedy=0):
        self.last_state = None
        self.lr = learning_rate  # 學習率
        self.gamma = reward_decay  # 折扣因子
        self.epsilon = e_greedy  # epsilon-greedy 策略的初始 epsilon
        self.q_table = {}  # Q 表，用字典表示
    
    def q_table_read(self, file_path):
        """
        從 CSV 文件讀取 Q 表
        """
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) == 8:  # 檢查行是否包含正確的數據
                    try:
                        state = tuple([int(row[0]), int(row[1]), int(row[2]), int(row[3])])
                        action_values = [float(row[4]), float(row[5]), float(row[6]), float(row[7])]
                        self.q_table[state] = action_values
                    except ValueError:
                        print("無效的行:", row)  # 輸出無效的行
                else:
                    print("無效的行:", row)  # 輸出無效的行
                    
    def q_table_save(self, file_path):
        """
        將 Q 表儲存為 CSV 文件
        """
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for state, action_values in self.q_table.items():
                try:
                    row = list(state) + action_values
                    writer.writerow(row)
                except TypeError as e:
                    print(f"Error writing row {state}: {e}")
                
                
    def choose_action(self, state):
        state_tuple = tuple(state)  # Convert state to tuple
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = [0, 0, 0, 0]  # Initialize Q values for the state: [右轉, 左轉, 前進, 發射]
        '''
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, len(self.q_table[state_tuple]) - 1)  # 隨機選擇動作
        else:
        '''
        max_value = max(self.q_table[state_tuple])
        max_indices = [i for i, v in enumerate(self.q_table[state_tuple]) if v == max_value]
        action = random.choice(max_indices)  # 選擇最大 Q 值的動作

        return action

    def learn(self, state, action, reward, next_state):
        state_tuple = tuple(state)  # Convert state to tuple
        next_state_tuple = tuple(next_state)  # Convert next_state to tuple

        # Ensure state and next_state exist in q_table
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = [0, 0, 0, 0]  # Initialize Q values for the state
        if next_state_tuple not in self.q_table:
            self.q_table[next_state_tuple] = [0, 0, 0, 0]  # Initialize Q values for the next state

        q_predict = self.q_table[state_tuple][action]  # Predicted Q value
        q_target = reward + self.gamma * max(self.q_table[next_state_tuple])  # Target Q value from Bellman equation
        self.q_table[state_tuple][action] += self.lr * (q_target - q_predict)  # Update Q value
        #print(f"q_table: {self.q_table[state_tuple][action]}")

        
        
    def state(self, scene_info, side):
        """
        Get the current state based on the scene information
        """
        my_angle = scene_info['angle']
        my_x = scene_info['x']
        my_y = scene_info['y']
        if scene_info['oil'] < 50:
            target = 0 #目標為油
            enemy_x = scene_info['oil_stations_info'][1]['x']
            enemy_y = scene_info['oil_stations_info'][1]['y']
            
        elif scene_info['power'] < 2:
            target = 0 #目標為子彈
            if side == "1P":
                enemy_x = scene_info['bullet_stations_info'][0]['x']
                enemy_y = scene_info['bullet_stations_info'][0]['y']
            else:    
                enemy_x = scene_info['bullet_stations_info'][1]['x']
                enemy_y = scene_info['bullet_stations_info'][1]['y']
            
        else:
            target = 1 #目標為敵人
            enemy_x = scene_info['competitor_info']['x']
            enemy_y = scene_info['competitor_info']['y']

        x_diff = my_x - enemy_x
        y_diff = my_y - enemy_y
        if my_angle ==135 or my_angle == 45:
            y_diff = y_diff+10
        if x_diff >0:
            x_diff = 1
        elif x_diff <0:
            x_diff = -1
        else:
            x_diff = 0
            
        if y_diff >10:
            y_diff = 1
        elif y_diff <-10:
            y_diff = -1
        else:
            y_diff = 0
            
        state = [y_diff, x_diff, target, my_angle]
        return state
        
    def reward(self, current_state):
        if self.last_state is None:
            self.last_state = current_state
            return 0
            
        reward = 0                
        last_state = self.last_state               
                        
        if current_state[0] > 0: 
            angle = 270
            if last_state[3] != angle:
                if abs(current_state[3] - angle) > abs(last_state[3] - angle): 
                    reward = 100 
                else:
                    reward = -100
            else:
                if last_state[3] == current_state[3]: 
                    reward = 100 
                else:
                    reward = -100
        elif current_state[0] < 0: 
            angle = 90
            if last_state[3] != angle:
                if abs(current_state[3] - angle) > abs(last_state[3] - angle): 
                    reward = 100 
                else:
                    reward = -100
            else:
                if last_state[3] == current_state[3]: 
                    reward = 100 
                else:
                    reward = -100
        else:
            angle = 180
            if last_state[3] != angle:
                if abs(current_state[3] - angle) > abs(last_state[3] - angle): 
                    reward = 100 
                else:
                    reward = -100
            else:
                if last_state[3] == current_state[3]: 
                    reward = 100 
                else:
                    reward = -100
 
        last_state = current_state
        return -reward
        
        
class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        """
        Constructor

        @param side A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side.
        """
        print("Initial Game ml script 1P")
        self.side = ai_name
        self.time = 0
        self.ql = QLearning()  # 創建 QLearning 對象
        self.previous_state = None
        self.previous_action = None

    def update(self, scene_info: dict, *args, **kwargs):
        """
        Generate the command according to the received scene information
        """
        # print(scene_info)
        if scene_info["status"] != "GAME_ALIVE":
            # 重置遊戲狀態
            self.previous_state = None
            self.previous_action = None
            # 從 CSV 文件讀取 Q 
            return "RESET"
        
        self.ql.q_table_read("q_table.csv")
            
          
        current_state = self.ql.state(scene_info, self.side)
        
        # 根據當前狀態選擇動作
        action_idx = 2
        action_idx = self.ql.choose_action(current_state)
        actions = ["TURN_RIGHT", "TURN_LEFT", "FORWARD","SHOOT"]
        
        my_angle = scene_info['angle']
        range_ = 10
        sw = 1
        if sw == 0:
            if current_state[0] > 0: 
                angle = 270
                if my_angle > angle:
                    action_idx = 0
                elif my_angle < angle:
                    action_idx = 1    
                else:
                    action_idx = 2
                    
            elif current_state[0] < 0:
                angle = 90
                if my_angle > angle:
                    action_idx = 0
                elif my_angle < angle:
                    action_idx = 1    
                else:
                    action_idx = 2
            else:    
                if current_state[1] < 0:
                    angle = 180
                else:
                    angle = 0
                
                if my_angle > angle:
                    action_idx = 0
                elif my_angle < angle:
                    action_idx = 1    
                else:
                    if current_state[2] == 1:
                        action_idx = 3
                    else:
                        action_idx = 2
                
                
        command = []
        command.append(actions[action_idx])
        if self.side == "3P":
            command = []
            command.append(actions[3])
               # 獎勵系統 (這裡可以根據具體情況修改)
        else:
            reward = self.ql.reward(current_state)

            # 在新狀態下學習
            if self.previous_state is not None and self.previous_action is not None:
                self.ql.learn(self.previous_state, self.previous_action, reward, current_state)

            #self.ql.q_table_save("q_table.csv")
            # 更新前一狀態和動作
            self.previous_state = current_state
            self.previous_action = action_idx

        return command


    def reset(self):
        """
        Reset the status
        """
        print(f"reset Game {self.side}")
