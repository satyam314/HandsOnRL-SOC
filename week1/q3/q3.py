from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def func(t, v, k):
    """ Computes the function S(t) with constants v and k. """
    return v * np.exp(-k * t)

def find_constants(df: pd.DataFrame, func: Callable):
    """ Fits a curve using SciPy to estimate constants v and k. """
    t = df['t'].values
    S = df['S'].values
    
    # Fit the curve to find v and k
    popt, _ = curve_fit(func, t, S)
    v, k = popt
    
    return v, k

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data.csv")
    
    # Find constants
    v, k = find_constants(df, func)
    v = round(v, 4)
    k = round(k, 4)
    print(v, k)
    
    # Plot histogram and fitted curve
    t = df['t'].values
    S = df['S'].values
    
    # Plot data points
    plt.scatter(t, S, label='Data', color='blue')
    
    # Plot fitted curve
    t_fit = np.linspace(min(t), max(t), 100)
    S_fit = func(t_fit, v, k)
    plt.plot(t_fit, S_fit, label=f'Fitted curve: v={v}, k={k}', color='red')
    
    # Labels and title
    plt.xlabel('t')
    plt.ylabel('S(t)')
    plt.title('Data and Fitted Curve')
    plt.legend()
    
    # Save plot
    plt.savefig('fit_curve.png')
    plt.show()
