import gym
import numpy as np
import pandas
from algorithms.q_learning import QLearningAgent

N_EPISODES = 40000

env = gym.make('CartPole-v0')


def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))


def to_bin(value, bins):
    return np.digitize(x=[value], bins=bins)[0]


def discretize():
    n_bins = 16
    n_bins_angle = 16

    cart_position = (env.observation_space.low[0], env.observation_space.high[0])
    cart_velocity = (env.observation_space.low[1], env.observation_space.high[1])
    pole_angle = (env.observation_space.low[2], env.observation_space.high[2])
    pole_angle_rate = (env.observation_space.low[3], env.observation_space.high[3])

    cart_position_bins = pandas.cut([cart_position[0], cart_position[1]], bins=n_bins, retbins=True)[1][1:-1]
    cart_velocity_bins = pandas.cut([cart_velocity[0], cart_velocity[1]], bins=n_bins, retbins=True)[1][1:-1]
    pole_angle_bins = pandas.cut([pole_angle[0], pole_angle[1]], bins=n_bins_angle, retbins=True)[1][1:-1]
    angle_rate_bins = pandas.cut([pole_angle_rate[0], pole_angle_rate[1]], bins=n_bins_angle, retbins=True)[1][1:-1]

    return cart_position_bins, cart_velocity_bins, pole_angle_bins, angle_rate_bins


def main():
    agent = QLearningAgent(actions=range(env.action_space.n),
                           alpha=0.1,
                           gamma=0.9,
                           epsilon=0.3)

    cart_position_bins, cart_velocity_bins, pole_angle_bins, angle_rate_bins = discretize()

    for i_episode in range(N_EPISODES):
        print("Episode:", i_episode)
        state = env.reset()
        done = False
        cumulative_reward = 0

        while not done:
            env.render()

            # choose an action
            state_id = build_state([
                to_bin(state[0], cart_position_bins),
                to_bin(state[1], cart_velocity_bins),
                to_bin(state[2], pole_angle_bins),
                to_bin(state[3], angle_rate_bins)
            ])
            action = agent.get_action(state_id)

            # perform the action
            state, reward, done, _ = env.step(action)

            next_state_id = build_state([
                to_bin(state[0], cart_position_bins),
                to_bin(state[1], cart_velocity_bins),
                to_bin(state[2], pole_angle_bins),
                to_bin(state[3], angle_rate_bins)
            ])

            cumulative_reward = cumulative_reward + reward
            if not done:
                # update q-learning agent
                agent.update(state_id, action, next_state_id, reward)
            else:
                reward = 0.0
                print("Reward: ", cumulative_reward)
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