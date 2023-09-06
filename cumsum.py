from typing import List, Union
import numpy as np

class CUSUM:
    """Cumulative Sum algorithm for detecting changepoints in univariate data streams.
    
    Read more in the original papers and literature about the algorithm.
    
    Parameters
    ----------
    k : float, default=0.25
        Control parameter for monitoring the gap between the normalized stream and the algorithm statistics.
    
    h : float, default=8.
        Control parameter used as a threshold for the decision rule.
    
    burnin : int, default=50
        Number of first observed values processed before a changepoint can be detected.
    
    mu : float, default=0.
        Initial mean value of the stream. Assumes that observations are normally distributed.
    
    sigma : float, default=1.
        Initial standard deviation of the stream.
    
    Attributes
    ----------
    _mu : List[float]
        Recording of all mean values.
    
    _sigma : List[float]
        Recording of all standard deviation values.
    
    S : List[float]
        S statistic calculated sequentially after processing each new observation which measures increases.
    
    T : List[float]
        T statistics calculated sequentially after processing each new observation which measures decreases.
    
    changepoints : List[int]
        List containing detected changepoints, initialized empty.
    
    n : int
        Number of observations in the current run length.
    """
    
    def __init__(self, k: float = 0.25, h: float = 8., burnin: int = 50, mu: float = 0., sigma: float = 1.):
        self.k = k
        self.h = h
        self.burnin = burnin
        self.initial_mu = mu
        self.initial_sigma = sigma

        self.reset()
    
    def update_mean_variance(self, data_new: float) -> None:
        """Properly update mean and variance using Welford's online algorithm."""
        n1 = self.n - 1
        delta = data_new - self.mu
        mu_new = self.mu + delta / self.n
        delta2 = data_new - mu_new
        m2 = n1 * self.sigma ** 2 + delta * delta2
        self.sigma = np.sqrt(m2 / self.n)
        self.mu = mu_new
        self._mu.append(mu_new)
        self._sigma.append(self.sigma)
        
    def update_statistics(self, data_new: float) -> None:
        self.S.append(max(0, self.S[-1] + (data_new - self.mu) / self.sigma - self.k))
        self.T.append(max(0, self.T[-1] - (data_new - self.mu) / self.sigma - self.k))
    
    def decision_rule(self, i: int) -> None:
        if (i >= self.burnin) and ((self.S[-1] > self.h) or (self.T[-1] > self.h)):
            self.changepoints.append(i)
            self.S[-1] = 0
            self.T[-1] = 0
            self.n = 2
        else:
            self.n += 1
        
    def process(self, data: Union[List[float], np.ndarray]) -> None:
        if not data:
            raise ValueError("Data stream should not be empty.")
        
        if type(data) not in [list, np.ndarray]:
            raise TypeError("Data stream should be a list or numpy array.")
        
        for i in range(1, len(data)):
            self.update_mean_variance(data[i])
            self.update_statistics(data[i])
            self.decision_rule(i)

    def reset(self) -> None:
        self.mu = self.initial_mu
        self.sigma = self.initial_sigma
        self._mu = [self.initial_mu]
        self._sigma = [self.initial_sigma]
        self.S = [0.]
        self.T = [0.]
        self.changepoints = []
        self.n = 2


if __name__ == "__main__":
    # Create an instance
    cusum_instance = CUSUM()
    
    cusum_instance.update_mean_variance(5.0)
    cusum_instance.update_statistics(5.0)
    
    print(cusum_instance.S[-1])
    assert cusum_instance.S[-1]==0.7122504486493763
    assert cusum_instance.T[-1]==0.0

    # Simulated data stream with a mean shift at index 100
    data_stream = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])

    # Process the data stream
    cusum_instance.process(data_stream.tolist())

    # Detected changepoints
    print("Detected Changepoints:", cusum_instance.changepoints)

    # Reset the instance
    cusum_instance.reset()
