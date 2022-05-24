# Source code: https://pythonprogramming.net/deep-q-learning-dqn-reinforcement-learning-python-tutorial/


import keras
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
# import tensorflow as tf

from collections import deque
import numpy as np
import gym
import random
from tqdm import tqdm


REPLAY_MEMORY_SIZE = 50_000 # Number of last steps to keep for model training 
MIN_REPLAY_MEMORY_SIZE = 1_000 # Min. number of steps in memory to start training
MINIBATCH_SIZE = 64 # How many steps/samples to use for training
DISCOUNT = 0.99 
UPDATE_TARGET_EVERY = 5 # Terminal states (end of episodes)
MIN_REWARD = -1
MEMORY_FRACTION = 0.20
EPISODES = 20_000 

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 50 # episodes
SHOW_PREVIEW = False





class DQNAgent:

    def __init__(self):
        
        # Main model
        ''' Gets .fit() -> trained at every step'''
        self.model = self.create_model()
        
        # Target model
        ''' Gets .predict() at every step
            After every 'n' steps the target model gets updated with main model's weights'''
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights()) # Receives the main model's weights
        self.target_update_counter = 0 # When the traget_update_counter hits a number then the weights are copied

        ''' The main model will apply .fit() for 1 value unless a batch of data is fed to it
            Take a random sampling of the replay memory to feed to the main model'''
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) 


    def create_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=env.observation_space.shape))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(nb_actions))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        ''' Parameter: transition which is the observation space, action reward and new values for each'''
        self.replay_memory.append(transition)

    
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]


    def train(self, terminal_state, step):
        
        ''' If the size of the replay memory is smaller than threshold value -> do nothing'''
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return 

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X=[]
        y=[]

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done: # We have not reached the terminal state
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)
        
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False if terminal_state else None)

        # Update to determine if we want to update target model
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY: # If it hits this value, update weights
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0 # Needs to be reset






##======================== Running ================================================

env = gym.make('CartPole-v0')
nb_actions = env.action_space.n

agent = DQNAgent()

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit='episode'):

    episode_reward = 0
    step = 1
    current_state = env.reset()

    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.action_space.n)
            
        
        new_state, reward, done, _ = env.step(action)

        episode_reward += reward 

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)




