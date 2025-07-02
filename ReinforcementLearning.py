import time
from collections import deque, namedtuple
import gym
import numpy
import PIL.Image
import tensorflow as tf
import utils
from pyvirtualdisplay import Display
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, losses, optimizers, regularizers

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
    #def __init__(self):

    def BuildModels(self):
        # Create the Q-Network
        self._q_network = Sequential([
            ### START CODE HERE ###
            Input(shape=(self._state_size)),
            Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
            Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
            Dense(self._num_actions, activation = 'linear')
            ### END CODE HERE ### 
            ])

        # Create the target Q^-Network
        self._target_q_network = Sequential([
            ### START CODE HERE ### 
            Input(shape=(self._state_size)),
            Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
            Dense(64, activation = 'relu', kernel_regularizer=regularizers.l2(0.1)),
            Dense(self._num_actions, activation = 'linear')
            ### END CODE HERE ###
            ])

        ### START CODE HERE ### 
        self._optimizer=tf.keras.optimizers.Adam(learning_rate=self.ALPHA)

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
        ### START CODE HERE ### 
        y_targets = rewards + (1 - done_vals) *  gamma * max_qsa
        ### END CODE HERE ###
        
        # Get the q_values and reshape to match y_targets
        q_values = self._q_network(states)
        q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                    tf.cast(actions, tf.int32)], axis=1))
            
        # Compute the loss
        ### START CODE HERE ### 
        loss = MSE(y_targets, q_values)
        ### END CODE HERE ### 
        return loss
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