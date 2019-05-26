import gym

from keras.layers import Input
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from algorithms.actor_critic import ActorCritic

env = gym.make('Pendulum-v0')

n_actions = env.action_space.shape[0]

ac = ActorCritic()

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=n_actions, theta=.15, mu=0., sigma=.3)
action_input = Input(shape=(n_actions,), name='action_input')

# create an agent and compile
agent = DDPGAgent(nb_actions=n_actions,
                  actor=ac.create_actor_model(env.observation_space.shape, n_actions),
                  critic=ac.create_critic_model(env.observation_space.shape, action_input),
                  critic_action_input=action_input,
                  memory=memory,
                  nb_steps_warmup_critic=100,
                  nb_steps_warmup_actor=100,
                  random_process=random_process,
                  gamma=.99,
                  target_model_update=1e-3)
agent.compile(Adam(lr=.01, clipnorm=1.), metrics=['mae'])

# train an agent
agent.fit(env, nb_steps=30000, visualize=False, verbose=1, nb_max_episode_steps=200)

# save an agent
agent.save_weights('ddpg_weights.h5f', overwrite=True)

# evaluate for 10 episodes
agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=200)
