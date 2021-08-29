import numpy as np
import torch
import torch.nn as nn
from network_utils import build_mlp, device, np2torch
import torch.functional as F
from general import batch_iterator

class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network
    """
    def __init__(self, env, config):
        """
        TODO:
        Create self.network using build_mlp, and create self.optimizer to
        optimize its parameters.
        You should find some values in the config, such as the number of layers,
        the size of the layers, and the learning rate.
        """
        super().__init__()
        self.config = config
        self.env = env
        self.baseline = None
        self.lr = self.config.learning_rate

        #######################################################
        #########   YOUR CODE HERE - 2-8 lines.   #############
        input_ = self.env.observation_space.shape[0]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # output_ = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        output_ = self.env.action_space.n
        self.network = build_mlp(input_, output_, config.n_layers, config.layer_size)
        self.network.to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        #######################################################
        #########          END YOUR CODE.          ############

    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]

        TODO:
        Run the network forward and then squeeze the result so that it's
        1-dimensional. Put the squeezed result in a variable called "output"
        (which will be returned).

        Note:
        A nn.Module's forward method will be invoked if you
        call it like a function, e.g. self(x) will call self.forward(x).
        When implementing other methods, you should use this instead of
        directly referencing the network (so that the shape is correct).
        """
        #######################################################
        #########   YOUR CODE HERE - 1 lines.     #############
        output = self.network(observations).flatten() # Can Flatten() as well
        print(output.shape)
        

        #######################################################
        #########          END YOUR CODE.          ############
        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]

        TODO:
        Evaluate the baseline and use the result to compute the advantages.
        Put the advantages in a variable called "advantages" (which will be
        returned).

        Note:
        The arguments and return value are numpy arrays. The np2torch function
        converts numpy arrays to torch tensors. You will have to convert the
        network output back to numpy, which can be done via the numpy() method.
        """
        observations = np2torch(observations)
        #######################################################
        #########   YOUR CODE HERE - 1-4 lines.   ############
        advantages = returns - self.forward(np2torch(observations)).cpu().data.numpy()

        #######################################################
        #########          END YOUR CODE.          ############
        return advantages

    def update_baseline(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size], containing all discounted
                future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]

        TODO:
        Compute the loss (MSE), backpropagate, and step self.optimizer.
        You may (though not necessary) find it useful to do perform these steps
        more than one once, since this method is only called once per policy update.
        If you want to use mini-batch SGD, we have provided a helper function
        called batch_iterator (implemented in general.py).
        """
        returns = np2torch(returns)
        observations = np2torch(observations)
        #######################################################
        #########   YOUR CODE HERE - 4-10 lines.  #############
        output = self.forward(observations)
        print(output.shape)
        print(returns.shape)
        for opt, ret in batch_iterator(output, returns): # Can also iterate over a batch over multiple epochs
            loss = F.MSELoss(ret, opt)
            loss.backward()
            self.optimizer.step()

        #######################################################
        #########          END YOUR CODE.          ############
