import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
from scipy.ndimage.interpolation import map_coordinates

BOARD_SIZE = 200
AGENT_COUNT = 120
ANTENNA_LEN = 4
ANGLE_TILT = 0.3
ANGLE_FOLLOW = 0.9
ANGLE_RANDOM = 0.4
PHEROMONE_DECAY_RATE = 0.995
ACTIVENESS_DECAY_RATE = 0.98
FOOD_DECAY_RATE = 0.01
INIT_FOOD_COUNT = 5

add_food_flg = 0

class Agent:
    def __init__(self, BOARD_SIZE, p1_board: np.ndarray, p2_board: np.ndarray, food_map: np.ndarray) -> None:
        self.BOARD_SIZE = BOARD_SIZE
        self.x = random.randint(1, 10)
        self.y = random.randint(1, 10)
        self.angle = random.random() * 10
        # self.vx, self.vy = self.norm(self.vx, self.vy)
        self.p1_board = p1_board
        self.p2_board = p2_board
        self.food_map = food_map
        self.is_empty = 1
        self.activeness = 0.1
    
    def dist(self, x, y):
        return math.sqrt(x**2 + y**2)
    
    def norm(self, x, y):
        d = self.dist(x, y)
        if d == 0:
            return x, y
        else:
            return x/d, y/d
    
    def get_interp_value(self, map, x, y):
        return map_coordinates(map, [[y], [x]], order=1)[0]
        # x = min(max(x, 0), BOARD_SIZE-1)
        # y = min(max(y, 0), BOARD_SIZE-1)
        # return map[round(x), round(y)]

    def load_unload_food(self, food_map):
        global add_food_flg
        if food_map[round(self.x), round(self.y)] >= FOOD_DECAY_RATE:
            if self.is_empty == 1:
                food_map[round(self.x), round(self.y)] -= FOOD_DECAY_RATE
                self.is_empty = -1
                self.activeness = 1
                self.angle += math.pi + random.random() * ANGLE_RANDOM - ANGLE_RANDOM/2
                if food_map[round(self.x), round(self.y)] <= FOOD_DECAY_RATE:
                    add_food_flg = 1
        elif self.x < 10 and self.y < 10:
            self.is_empty = 1
            self.activeness = 1
            self.angle -= random.random() * ANGLE_RANDOM - ANGLE_RANDOM/2

    def strategy(self):
        if self.is_empty == 1:
            board = self.p2_board
        else:
            board = self.p1_board
        # front = self.y + math.sin(self.angle), self.x + math.cos(self.angle)
        left  = self.y + math.sin(self.angle-ANGLE_TILT) * ANTENNA_LEN, self.x + math.cos(self.angle-ANGLE_TILT) * ANTENNA_LEN
        right = self.y + math.sin(self.angle+ANGLE_TILT) * ANTENNA_LEN, self.x + math.cos(self.angle+ANGLE_TILT) * ANTENNA_LEN
        # front_v = self.get_interp_value(board, front[0], front[1])
        left_v  = self.get_interp_value(board, left[0], left[1])
        right_v = self.get_interp_value(board, right[0], right[1])
        sum_v = (left_v + right_v)
        min_v = min(left_v, right_v)
        angle_delta = random.random() * ANGLE_RANDOM - ANGLE_RANDOM/2
        if sum_v < 1e-3:
            self.angle += angle_delta
            return
        if random.random() < 0.1 and min_v > 1e-3:
            angle_delta += (ANGLE_TILT * left_v - ANGLE_TILT * right_v) * ANGLE_FOLLOW / sum_v
        else:
            angle_delta += (-ANGLE_TILT * left_v + ANGLE_TILT * right_v) * ANGLE_FOLLOW / sum_v
        self.angle += angle_delta
        # # print(left_v, front_v, right_v)
        # if max_v == front_v:
        #     return
        # elif max_v == left_v:
        #     self.angle -= (ANGLE_FOLLOW * ANGLE_TILT)
        # elif max_v == right_v:
        #     self.angle += (ANGLE_FOLLOW * ANGLE_TILT)
        # self.angle += random.random() * ANGLE_RANDOM - ANGLE_RANDOM/2

        # front_v += 1-max_v
        # left_v += 1-max_v
        # right_v += 1-max_v
        # sum_v = (front_v+left_v+right_v)
        # new_angle = self.angle - ANGLE_TILT*(left_v / sum_v) + ANGLE_TILT*(right_v / sum_v)
        # new_angle += random.random() * 0.02 - 0.01
        # self.angle = new_angle

    def pheromone(self):
        if self.is_empty == 1:
            self.p1_board[round(self.x), round(self.y)] += self.activeness
        else:
            self.p2_board[round(self.x), round(self.y)] += self.activeness

    def update(self):
        # strategy
        self.strategy()

        x_new = self.x + math.cos(self.angle)
        y_new = self.y + math.sin(self.angle)

        while x_new < 1 or y_new < 1 or x_new >= self.BOARD_SIZE-1 or y_new >= self.BOARD_SIZE-1:
            self.angle += random.random() * 10
            # self.angle += math.pi
            x_new = self.x + math.cos(self.angle)
            y_new = self.y + math.sin(self.angle)
        
        self.x = x_new
        self.y = y_new

        self.load_unload_food(self.food_map)

        self.pheromone()
        self.activeness *= ACTIVENESS_DECAY_RATE
        # self.activeness -= 1-ACTIVENESS_DECAY_RATE
        

my_board = np.zeros((3, BOARD_SIZE, BOARD_SIZE))
my_p1_board = my_board[0]
my_p2_board = my_board[1]
my_food_map = my_board[2]
for i in range(INIT_FOOD_COUNT):
    my_food_map[random.randint(50, BOARD_SIZE-5), random.randint(50, BOARD_SIZE-5)] = 1

agents = []
for i in range(AGENT_COUNT):
    agents.append(Agent(BOARD_SIZE, my_p1_board, my_p2_board, my_food_map))

xs = [agent.x for agent in agents]
ys = [agent.y for agent in agents]

fig, axs = plt.subplots()
# ag_scatter = axs.scatter(ys, xs, c='w', marker='.')
im = axs.imshow(np.clip(np.transpose(my_board, axes=(1,2,0)), 0, 1))

def update_board(ret):
    global my_p1_board, my_p2_board
    for agent in agents:
        agent.update()
    my_p1_board *= PHEROMONE_DECAY_RATE
    my_p2_board *= PHEROMONE_DECAY_RATE
    if ret is True:
        agent_positions = [[agent.y, agent.x] for agent in agents]
        # xs = [agent.x for agent in agents]
        # ys = [agent.y for agent in agents]
        k = np.clip(np.transpose(my_board, axes=(1,2,0)), 0, 1)
        k[:,:,0] += k[:,:,2]
        # k[:,:,1] += k[:,:,2]
        k = np.clip(k, 0, 1)
        return k, agent_positions



def animate(frame):
    global add_food_flg
    for i in range(10):
        update_board(False)
    img, agent_positions = update_board(True)
    im.set_data(img)
    # ag_scatter.set_offsets(agent_positions)
    if add_food_flg == 1:
        add_food_flg = 0
        my_food_map[random.randint(50, BOARD_SIZE-5), random.randint(50, BOARD_SIZE-5)] = 1
        print('added food')
    return im,

anim = animation.FuncAnimation(fig, animate, frames=200,
                               interval=5)
                               
plt.show()