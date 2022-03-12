# -*- coding: utf-8 -*-
# @File    : 2_ii.py
# @Author  : Zichi Zhang
# @Date    : 2022/3/11
# @Software: PyCharm
import pandas as pd
x = pd.read_csv('exponential_data.csv', header=None)

num_samples = len(x)

E_x = x.mean()
print(E_x)
lambda_star = 1/E_x
print(lambda_star)


