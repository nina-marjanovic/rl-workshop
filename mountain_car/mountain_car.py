import gym
import numpy as np
import pandas
from algorithms.q_learning import QLearningAgent

N_EPISODES = 30000

env = gym.make('MountainCar-v0')


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


def discretize():
    n_bins = 50

    cart_position = (env.observation_space.low[0], env.observation_space.high[0])
    cart_velocity = (env.observation_space.low[1], env.observation_space.high[1])

    cart_position_bins = pandas.cut([cart_position[0], cart_position[1]], bins=n_bins, retbins=True)[1]
    cart_velocity_bins = pandas.cut([cart_velocity[0], cart_velocity[1]], bins=n_bins, retbins=True)[1]

    return cart_position_bins, cart_velocity_bins


def main():
    agent = QLearningAgent(actions=range(env.action_space.n),
                           alpha=0.1,
                           gamma=0.9,
                           epsilon=0.1)

    cart_position_bins, cart_velocity_bins = discretize()

    for i_episode in range(N_EPISODES):
        print("Episode:", i_episode)
        state = env.reset()
        done = False
        positions = []  # used for finding the max position in an episode

        while not done:
            if i_episode > 13000:   # at this point the problem should be solved.
                                    # we render only these solutions because it makes
                                    # training faster
                env.render()

            # choose an action
            state_id = build_state([to_bin(state[0], cart_position_bins),
                                    to_bin(state[1], cart_velocity_bins)])
            action = agent.get_action(state_id)

            # perform the action
            state, reward, done, _ = env.step(action)
            positions.append(state[0])

            next_state_id = build_state([to_bin(state[0], cart_position_bins),
                                         to_bin(state[1], cart_velocity_bins)])

            if not done:
                # update the q-learning agent
                agent.update(state_id, action, next_state_id, reward)
            else:
                print("Max position: ", max(positions))
                reward = 0.0
                agent.update(state_id, action, next_state_id, reward)
                break

    env.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            env.close()
        except SystemExit:
            print("error")