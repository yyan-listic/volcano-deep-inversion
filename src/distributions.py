from src.importation import special, numpy as np

"""
my own distributions because scipy is wrong for some

if you want to add your owns, inherit from Distribution, follow its documentation
and add it to distributions_table
"""

class Distribution:

    def __init__(self) -> None:
        """
        add parameters that define your distribution
        """
        pass
    
    def qf(self, q: float) -> float:
        """
        give value of variable following this distribution from its quantile q

        -q: (float) a quantile value (between 0 and 1)

        return: (float)
        """
        pass

class Log_Normal(Distribution):
    
    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma
    
    def qf(self, q: float) -> float:
        return np.exp(self.mu + np.sqrt(2) * self.sigma * special.erfinv(2 * q - 1))

class Uniform(Distribution):
    
    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b
    
    def qf(self, q: float) -> float:
        return (self.b - self.a) * q + self.a

class Exponential(Distribution):
    
    def __init__(self, l: float) -> None:
        self.l = l
    
    def qf(self, q: float) -> float:
        return - np.log(1 - q) / self.l

distributions_table = {
    "log_normal": Log_Normal,
    "uniform": Uniform,
    "exponential": Exponential
}