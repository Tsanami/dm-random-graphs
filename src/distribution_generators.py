import numpy as np

def generate_h0(n, mu=0, sigma=1): # временно
    return np.random.normal(mu, sigma, n)

def generate_h1(n, lam=1): # временно
    return np.random.exponential(1/lam, n)