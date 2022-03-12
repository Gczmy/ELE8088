# -*- coding: utf-8 -*-
# @File    : 2_vi.py
# @Author  : Zichi Zhang
# @Date    : 2022/3/12
# @Software: PyCharm
import pandas as pd
x = pd.read_csv('normal_data.csv', header=None)

num_samples = len(x)

mu = x.mean()
print(mu)
sigma2 = x.var()
print(sigma2)
