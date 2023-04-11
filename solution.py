import pandas as pd
import numpy as np
from scipy.stats import t

chat_id = 123456 # Ваш chat ID, не меняйте название переменной

def solution(control: np.array, test: np.array) -> bool: # Одна или две выборке на входе, заполняется исходя из условия
    if control.size == 0 or test.size == 0:
        return False
    x1 = np.mean(control)
    x2 = np.mean(test)
    s1 = np.std(control, ddof=1)
    s2 = np.std(test, ddof=1)
    n1 = control.size
    n2 = test.size
    t_stat = (x1 - x2) / np.sqrt(s1**2/n1 + s2**2/n2)
    alpha = 0.07
    df = n1 + n2 - 2
    t_alpha = t.ppf(1 - alpha/2, df)
    return t_stat > t_alpha
