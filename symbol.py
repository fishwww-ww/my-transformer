import pandas as pd
import numpy as np

def handle_symbol(pvo, symbol):
    data = {}
    for p, s in zip(pvo, symbol):
        # 确保 s 是一个标量
        s = s.item() if isinstance(s, np.ndarray) else s
        if s not in data:
            data[s] = []
        data[s].append(p)
    return data
