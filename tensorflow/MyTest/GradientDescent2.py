import numpy as np

# 2개의 변수를 가지는 함수의 최소값은?

def f(x):
    return 2*x[0]**2 + 2*x[0]*x[1] + x[1]**2

def df(x):
    dx = 4*x[0] + 2*x[1]
    dy = 2*x[0] + 2*x[1]
    return np.array([dx, dy])

rho = 0.005
precision = 0.00000001
difference = 100
x = np.random.rand(2)

while difference > precision:
    dr = df(x)
    prev_x = x
    x = x - rho * dr
    difference = np.dot(x-prev_x, x-prev_x)
    print("x = {}, df = {}, f(x) = {:f}"
          .format(np.array2string(x), np.array2string(dr), f(x)))