import random
import numpy as np


class QLearningAgent:

    def __init__(self, actions, epsilon=0.1, gamma=0.8, alpha=0.1):
        self.alpha = alpha                  # learning rate
        self.gamma = gamma                  # discount factor
        self.epsilon = epsilon              # exploration probability
        self.actions = actions              # list of actions
        self.qs = {}                        # q (state) table

    def get_q_value(self, state, action):
        if not (state in self.qs) or not (action in self.qs[state]):
            return 0.0
        else:
            return self.qs[state][action]

    def get_action(self, state):
        """
        Returns the best action for the current state
        """
        q = [self.get_q_value(state, action) for action in self.actions]

        # exploration
        if random.random() <= self.epsilon:
            return self.actions[random.randint(0, len(self.actions)-1)]
        return self.actions[np.argmax(q)]

    def update(self, state, action, next_state, reward):
        """
        Update q-value of the given state
        """
        if state not in self.qs:
            self.qs[state] = {}
        if action not in self.qs[state]:
            self.qs[state][action] = reward
        else:
            new_max_q = max([self.get_q_value(next_state, a) for a in self.actions])
            self.qs[state][action] += self.alpha * (reward + self.gamma * new_max_q - self.qs[state][action])
