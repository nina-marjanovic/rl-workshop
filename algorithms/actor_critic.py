from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate


class ActorCritic:

    def create_actor_model(self, n_observation_space, n_actions):
        actor = Sequential()
        actor.add(Flatten(input_shape=(1,) + n_observation_space))
        actor.add(Dense(16))
        actor.add(Activation('relu'))
        actor.add(Dense(16))
        actor.add(Activation('relu'))
        actor.add(Dense(16))
        actor.add(Activation('relu'))
        actor.add(Dense(n_actions))
        actor.add(Activation('linear'))
        print(actor.summary())
        return actor

    def create_critic_model(self, n_observation_space, action_input):
        observation_input = Input(shape=(1,) + n_observation_space,
                                  name='observation_input')
        flattened_observation = Flatten()(observation_input)
        x = Concatenate()([action_input, flattened_observation])
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('linear')(x)
        critic = Model(inputs=[action_input, observation_input], outputs=x)
        print(critic.summary())
        return critic
