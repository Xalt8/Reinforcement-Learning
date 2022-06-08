import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import count
from matplotlib.animation import FuncAnimation

# gp = (x for x in range(10,15))
# reward = 0

# def done(reward):
#     #observation, reward, done, info
#     done = False
#     try:
#         observation = next(gp)
#         reward += observation
#         return observation, reward, done
#     except:
#         done = True
#         return observation, reward, done
    
# for i in range(5):
#     print(done(reward))

# Not used
def get_neighbours(index:tuple, neighbourhood:str) -> list[tuple]:
    ''' Source: https://fastlife.readthedocs.io/en/latest/notes/neighborhoods.html '''
    # goal = 5,5
    assert neighbourhood == 'von_neumann' or neighbourhood == 'moore', "neighbourhood should be 'von_neumann' or 'moore'"  

    index_offset = {'von_neumann': [(-1,0), (0,1), (1,0),(0,-1)],
                    'moore': [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1, -1), (0,-1), (-1,-1)]} 

    indices = [(index[0] + row, index[1] + col) for row, col in index_offset[neighbourhood]]
    return indices



def wavefront(index:tuple, matrix:np.ndarray)-> np.ndarray:
    ''' Uses Depth First approach to make an array with values that that concentrically 
        increase from a given index tuple'''

    assert index[0] < matrix.shape[0] and index[1] < matrix.shape[1], f"The index values need to be <= {matrix.shape}"
    
    zero_mat = np.zeros(matrix.shape)
    unvisited = [(row, col) for row in np.arange(matrix.shape[0]) for col in np.arange(matrix.shape[1])] 
    
    offset_1 = 1 # Get 1 less than the index value
    offset_2 = 2 # Get 1 more than the index value
    val = 1

    while unvisited:
        for row in range(index[0] - offset_1, index[0] + offset_2):
            for col in range(index[1] - offset_1, index[1] + offset_2):
                if 0 <= row < matrix.shape[0] \
                    and 0 <= col <matrix.shape[1] \
                    and (row, col) in unvisited:
                    zero_mat[row][col] = val
                    unvisited.remove((row, col))
        offset_1 += 1
        offset_2 += 1
        val += 1

    zero_mat[index] = 0 # Set the index to zero
    return zero_mat


def calc_reward(goal:tuple, position:tuple) -> float:
    ''' Takes the goal and a given position and calculates the distance
        and returns a rewards value '''
    return -np.sum(np.square(np.asarray(goal) - np.asarray(position)))                 



def test():
    x_vals = []
    y_vals = []

    index = count()

    def animate(i):
        x_vals.append(next(index))
        y_vals.append(random.randint(0,5))

        plt.cla()
        plt.plot(x_vals, y_vals)

    anim = FuncAnimation(plt.gcf(), animate, interval=1000)

    plt.tight_layout()
    plt.show()


def optimise() -> tuple:
    goal = np.array([5,5])
    pos = np.array([25,25]) # Start position explicit -> can be random
    
    npop = 50 # population size
    sigma = 2 # noise standard deviation -> 0.1
    alpha = 1 # learning rate -> 0.001
    
    for i in range(50):
        pop = np.random.randn(npop, goal.size)
        jittered_pop = np.array([pos + sigma * p for p in pop])
        rewards_array = np.array([calc_reward(goal=goal, position=jit) for jit in jittered_pop])
        std_rewards = (rewards_array - np.mean(rewards_array))/np.std(rewards_array)
        pos = pos + alpha/(npop*sigma) * np.dot(jittered_pop.T, std_rewards)

        yield pos, jittered_pop


def init():
    jits.set_offsets(np.empty(2))
    new_po.set_offsets(np.empty(2))


def animate(i) -> None:
    new_pos, jittered_pop = next(optimise())
    
    jits.set_offsets(jittered_pop)
    new_po.set_offsets(new_pos)
    
    return new_po, jits


if __name__ =='__main__':
    
    fig, ax = plt.subplots()
    m = wavefront(index=(5,5), matrix=np.arange(900).reshape(30,30))
    im = ax.imshow(m)
    jits = ax.scatter(x=[], y=[], c='red', edgecolors='black')
    new_po = ax.scatter(x=[], y=[], c='white', edgecolors='white')

    anim = FuncAnimation(fig, animate, frames=50, interval=200, blit=True)

    fig.tight_layout()
    plt.show()
    