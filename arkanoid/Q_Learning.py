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

# 使用範例
if __name__ == "__main__":
    q_learning = QLearning()
    
    # 模擬場景
    scene_info = {
        "frame": 0,
        "status": "GAME_ALIVE",
        "ball": [93, 395],
        "ball_served": False,
        "platform": [75, 400],
        "bricks": [[50, 60], [125, 80]],
        "hard_bricks": [[35, 50], [135, 90]]
    }

    state = q_learning.state(scene_info)
    action = q_learning.choose_action(state)
    print(f"Selected Action: {action}")

    # 存取Q表
    q_learning.q_table_save('q_table.csv')
    q_learning.q_table_read('q_table.csv')

    # 學習更新Q值
    next_state = q_learning.state(scene_info)
    q_learning.learn(state, action, reward=1, next_state=next_state)