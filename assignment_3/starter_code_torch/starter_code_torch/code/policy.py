import torch
import torch.nn as nn
import torch.distributions as ptd

from network_utils import np2torch, device


class BasePolicy:
    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution

        See https://pytorch.org/docs/stable/distributions.html#distribution

        This is an abstract method and must be overridden by subclasses.
        It will return an object representing the policy's conditional
        distribution(s) given the observations. The distribution will have a
        batch shape matching that of observations, to allow for a different
        distribution for each observation in the batch.
        """
        raise NotImplementedError

    def act(self, observations):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]

        TODO:
        Call self.action_distribution to get the distribution over actions,
        then sample from that distribution. You will have to convert the
        actions to a numpy array, via numpy(). Put the result in a variable
        called sampled_actions (which will be returned).
        """
        observations = np2torch(observations)
        #######################################################
        #########   YOUR CODE HERE - 1-3 lines.    ############
        sampled_actions = self.action_distribution(observations).sample().cpu().data.numpy()

        #######################################################
        #########          END YOUR CODE.          ############
        return sampled_actions


class CategoricalPolicy(BasePolicy, nn.Module):
    def __init__(self, network):
        nn.Module.__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = network
        self.network.to(self.device)

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network

        See https://pytorch.org/docs/stable/distributions.html#categorical
        """
        #######################################################
        #########   YOUR CODE HERE - 1-2 lines.    ############
        distribution = ptd.categorical.Categorical(logits=self.network(observations))
        #######################################################
        #########          END YOUR CODE.          ############
        return distribution


class GaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, network, action_dim):
        """
        After the basic initialization, you should create a nn.Parameter of
        shape [dim(action space)] and assign it to self.log_std.
        A reasonable initial value for log_std is 0 (corresponding to an
        initial std of 1), but you are welcome to try different values.
        """
        nn.Module.__init__(self)
        self.network = network
        #######################################################
        #########   YOUR CODE HERE - 1 line.       ############
        self.log_std = nn.Parameter(torch.tensor([0]*action_dim))
        # What is this exactly?

        #######################################################
        #########          END YOUR CODE.          ############

    def std(self):
        """
        Returns:
            std: torch.Tensor of shape [dim(action space)]

        The return value contains the standard deviations for each dimension
        of the policy's actions. It can be computed from self.log_std
        """
        #######################################################
        #########   YOUR CODE HERE - 1 line.       ############
        std = torch.exp(self.log_std)

        #######################################################
        #########          END YOUR CODE.          ############
        return std

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: an instance of a subclass of
                torch.distributions.Distribution representing a diagonal
                Gaussian distribution whose mean (loc) is computed by
                self.network and standard deviation (scale) is self.std()

        Note: PyTorch doesn't have a diagonal Gaussian built in, but you can
            fashion one out of
            (a) torch.distributions.MultivariateNormal
            or
            (b) A combination of torch.distributions.Normal
                             and torch.distributions.Independent
        """
        #######################################################
        #########   YOUR CODE HERE - 2-4 lines.    ############
        mean = self.network(observations) # Dimension won't be right
        distribution = ptd.multivariate_normal.MultivariateNormal(mean, self.std())

        #######################################################
        #########          END YOUR CODE.          ############
        return distribution