import gymnasium as gym
import tensorflow as tf
import utils, time, PIL.Image
from collections import deque, namedtuple
from pyvirtualdisplay import Display
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, losses, optimizers, regularizers

# XXX: I have NOT tested this file because installation of swig gymnasium[box2d] failed. Will return to this when I have the time. 
class ReinforcementLearning():
    """
    Deep Q-Learning with Experience Replay
    Initialize Q-Network with random weights w
    Initialize Q-target Network with weights w~ = w
    for episode i = 1 to M do:
      Receive initial observation state S1
      for t = 1 to T do:
        Observe state St and choose action At using an epsilon-greedy policy
        Take action At in the environment, receive reward Rt and next state S(t+1)
        Store experience tuple (St, At, Rt, St+1) in memory buffer D
        Every C steps perform a learning update:
          Sample random min-batch of experience typles (Sj,Aj,R,Sj+t) from D
          Set Yj = Rj if episode terminates at step j + 1. Otherwise set Yj = Rj + gamma * max(Q-target(Sj+1, a')) <- compute_loss() & agent_learn()
          Perform a gradient descent step on (Yj - Q(Sj,Aj;w)) ** 2 with respect to the Q-Network weights w <- agent_learn()
          Update the weights of the Q-target Network using a soft-update <- agent_learn()
      end
    end
    """
    MEMORY_SIZE = 100_000     # size of memory buffer
    GAMMA = 0.995             # discount factor
    ALPHA = 1e-3              # learning rate  
    NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps    
    _state_size: int = None
    _num_actions: int = None
    _q_network: Sequential = None
    _target_q_network: Sequential = None
    _optimizer = None
    _env = None
    def __init__(self):
        self._env = gym.make('LunarLander-v3')
        self._env.reset()
        PIL.Image.fromarray(self._env.render(mode='rgb_array'))
        self._state_size = self._env.observation_space.shape
        self._num_actions = self._env.action_space.n
        print('State Shape:', self._state_size)
        print('Number of actions:', self._num_actions)

    def BuildModels(self):
        # Create the Q-Network
        self._q_network = Sequential([
            Input(shape=(self._state_size)),
            Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)), # Densely connected, or fully connected
            Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
            Dense(self._num_actions, activation = 'linear')
            ])

        # Create the target Q^-Network
        self._target_q_network = Sequential([
            Input(shape=(self._state_size)),
            Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
            Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
            Dense(self._num_actions, activation = 'linear')
            ])

        self._optimizer=tf.keras.optimizers.Adam(learning_rate=self.ALPHA) # Intelligent gradient descent which automatically adjusts the learning rate (alpha) depending on the direction of the gradient descent.

    def compute_loss(self, experiences, gamma):
        """ 
        Calculates the loss.
        
        Args:
        experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
        gamma: (float) The discount factor.
        q_network: (tf.keras.Sequential) Keras model for predicting the q_values
        target_q_network: (tf.keras.Sequential) Keras model for predicting the targets
            
        Returns:
        loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
                the y targets and the Q(s,a) values.

        Implement line indicated above in the docstring. Also compute the loss between the Y targets and the Q(s,a) values. 
        Setting the Y targets equal to:
        Y = Rj (if episode terminates at step j + 1)
        Y = Rj + gamma * max(Q(Sj+1, a')) otherwise

        Here are a couple of things to note:

        The compute_loss function takes in a mini-batch of experience tuples. This mini-batch of experience tuples is unpacked to extract the states, actions, rewards, next_states, and done_vals. 
        You should keep in mind that these variables are TensorFlow Tensors whose size will depend on the mini-batch size. For example, if the mini-batch size is 64 then both rewards and done_vals will be TensorFlow Tensors with 64 elements.
        Using if/else statements to set the  Y targets will not work when the variables are tensors with many elements. However, notice that you can use the done_vals to implement the above in a single line of code. 
        To do this, recall that the done variable is a Boolean variable that takes the value True when an episode terminates at step j+1
        and it is False otherwise. Taking into account that a Boolean value of True has the numerical value of 1 and a Boolean value of False has the numerical value of 0, you can use the factor (1 - done_vals) to implement the above in a single line of code. Here's a hint: notice that (1 - done_vals) has a value of 0 when done_vals is True and a value of 1 when done_vals is False.
        Lastly, compute the loss by calculating the Mean-Squared Error (MSE) between the y_targets and the q_values. To calculate the mean-squared error you should use the already imported package MSE:                
        """

        # Unpack the mini-batch of experience tuples
        states, actions, rewards, next_states, done_vals = experiences
        
        # Compute max Q^(s,a)
        max_qsa = tf.reduce_max(self._target_q_network(next_states), axis=-1)
        
        # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
        y_targets = rewards + (1 - done_vals) *  gamma * max_qsa
        
        # Get the q_values and reshape to match y_targets
        q_values = self._q_network(states)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                    tf.cast(actions, tf.int32)], axis=1))
            
        # Compute the loss
        return MSE(y_targets, q_values)
    # @tf.function decorator to increase performance. Without this decorator our training will take twice as long. If you would like to know more about how to increase performance with @tf.function take a look at the TensorFlow documentation.
    @tf.function
    def agent_learn(self, experiences, gamma):
        """
        Updates the weights of the Q networks.
        
        Args:
        experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
        gamma: (float) The discount factor.
        """
        # Calculate the loss
        with tf.GradientTape() as tape:
            loss = self.compute_loss(experiences, gamma, self._q_network, self._target_q_network)

        # Get the gradients of the loss with respect to the weights.
        gradients = tape.gradient(loss, self._q_network.trainable_variables)
        
        # Update the weights of the q_network.
        self._optimizer.apply_gradients(zip(gradients, self._q_network.trainable_variables))

        # update the weights of target q_network
        utils.update_target_network(self._q_network, self._target_q_network)

def TestReinforcementLearning():
    reinforcement = ReinforcementLearning()
    reinforcement.BuildModels()
    start = time.time()

    num_episodes = 2000
    max_num_timesteps = 1000

    total_point_history = []

    num_p_av = 100    # number of total points to use for averaging
    epsilon = 1.0     # initial ε value for ε-greedy policy

    # Create a memory buffer D with capacity N
    memory_buffer = deque(maxlen=MEMORY_SIZE)

    # Set the target network weights equal to the Q-Network weights
    target_q_network.set_weights(q_network.get_weights())

    for i in range(num_episodes):
        
        # Reset the environment to the initial state and get the initial state
        state = env.reset()
        total_points = 0
        
        for t in range(max_num_timesteps):
            
            # From the current state S choose an action A using an ε-greedy policy
            state_qn = np.expand_dims(state, axis=0)  # state needs to be the right shape for the q_network
            q_values = q_network(state_qn)
            action = utils.get_action(q_values, epsilon)
            
            # Take action A and receive reward R and the next state S'
            next_state, reward, done, _ = env.step(action)
            
            # Store experience tuple (S,A,R,S') in the memory buffer.
            # We store the done variable as well for convenience.
            memory_buffer.append(experience(state, action, reward, next_state, done))
            
            # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
            update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)
            
            if update:
                # Sample random mini-batch of experience tuples (S,A,R,S') from D
                experiences = utils.get_experiences(memory_buffer)
                
                # Set the y targets, perform a gradient descent step,
                # and update the network weights.
                agent_learn(experiences, GAMMA)
            
            state = next_state.copy()
            total_points += reward
            
            if done:
                break
                
        total_point_history.append(total_points)
        av_latest_points = np.mean(total_point_history[-num_p_av:])
        
        # Update the ε value
        epsilon = utils.get_new_eps(epsilon)

        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

        if (i+1) % num_p_av == 0:
            print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

        # We will consider that the environment is solved if we get an
        # average of 200 points in the last 100 episodes.
        if av_latest_points >= 200.0:
            print(f"\n\nEnvironment solved in {i+1} episodes!")
            q_network.save('lunar_lander_model.h5')
            break
            
    tot_time = time.time() - start

    print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")
if __name__ == "__main__":
    TestReinforcementLearning()
