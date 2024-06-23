import numpy as np
import pandas as pd
import random

class QLearning:
    def __init__(self, eGreedy=0.1, learning_rate=0.7, discount_factor=0.9):
        self.eGreedy = eGreedy
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}

    def choose_action(self, state):
        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]  # [左移, 不動, 右移]
        
        if random.uniform(0, 1) < self.eGreedy:
            action = random.randint(0, 2)
        else:
            action = np.argmax(self.q_table[state])
        
        return action

    def q_table_save(self, path):
        q_table_df = pd.DataFrame.from_dict(self.q_table, orient='index', columns=['MOVE_LEFT', 'STAY', 'MOVE_RIGHT'])
        q_table_df.to_csv(path, index=True, header=True)
    
    def q_table_read(self, path):
        q_table_df = pd.read_csv(path, index_col=0)
        self.q_table = q_table_df.to_dict('index')

    def learn(self, state, action, reward, next_state):
        state = tuple(state)
        next_state = tuple(next_state)
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0, 0]

        predict = self.q_table[state][action]
        target = reward + self.discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (target - predict)

    def state(self, scene_info):
        ball_x, ball_y = scene_info["ball"]
        platform_x = scene_info["platform"][0]
        closest_brick_x = min([brick[0] for brick in scene_info["bricks"]], key=lambda x: abs(x - ball_x))

        ball_angle = np.arctan2(scene_info["ball"][1] - ball_y, scene_info["ball"][0] - ball_x)
        ball_platform_diff = ball_x - platform_x

        return [closest_brick_x, ball_platform_diff, ball_angle]
        
    def reward(self, state, state_, action, scene_info):
        ball_x, ball_y = state_[0], state_[1]
        platform_x, platform_y = scene_info["platform"][0], scene_info["platform"][1]
        bricks = scene_info["bricks"]
        reward = 0

        # 球和磚塊的座標差
        closest_brick_x = min([brick[0] for brick in bricks], key=lambda x: abs(x - ball_x))
        ball_platform_diff = ball_x - platform_x

        # 計算球的移動角度
        ball_angle = np.arctan2(state_[1] - state[1], state_[0] - state[0])

        # 判斷條件
        if ball_y == platform_y and platform_x <= ball_x <= platform_x + 40:  # 假設板子的寬度是40
            reward += 5
        if scene_info["status"] == "GAME_ALIVE":
          reward += 1
        if np.isclose(ball_angle, ball_platform_diff, atol=0.1):  # 假設需要考慮角度的誤差範圍
          reward += 10
        if platform_x <= ball_x <= platform_x + 40:
            reward += 5

        if ball_y > platform_y:
            reward -= 10
        if ball_x < platform_x or ball_x > platform_x + 40:
            reward -= 5

        return reward

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.ai_name = ai_name
        self.q_learning = QLearning()
        self.ball_served = False

    def update(self, scene_info, *args, **kwargs):
        q_learning = QLearning()
        
        if  'state' not in globals() or state is None:
            state = self.q_learning.state(scene_info) 
        if 'action' not in globals() or action is None:
            action = 1
            

        if scene_info["status"] in ["GAME_OVER", "GAME_PASS"]:
            return "RESET"
        #訓練模型
        state_ = self.q_learning.state(scene_info)        
        reward = self.q_learning.reward(state, state_, action, scene_info)
        q_learning.learn(state, action, reward, next_state=state_)
        state=state_
        
        print(f"Selected Action: {action}")
        

        
        if not self.ball_served:
            command = "SERVE_TO_LEFT"
            self.ball_served = True
        else:
            action = self.q_learning.choose_action(state)
            if action == 0:
                command = "MOVE_LEFT"
            elif action == 1:
                command = "STAY"
            else:
                command = "MOVE_RIGHT" # action

        return command

    def reset(self):
        self.ball_served = False