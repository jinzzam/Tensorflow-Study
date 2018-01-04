from random import random

def f(x):
    return x**4 - 12.0*x**2 - 12.0*x

def df(x):
    return 4 * x**3 - 24 * x - 12

rho = 0.005
precision = 0.000000001
difference = 100
x = random()
# x = 4.0

while difference > precision:
    dr = df(x)
    prev_x = x
    x = x - rho * dr
    difference = abs(prev_x - x)
    print("x = {:f}, df = {:10.6f}, f(x) = {:f}".format(x, dr, f(x)))


#Gradient descent 알고리즘은 local minimum에 빠질 수 있다.
#즉 항상 최소값을 찾는 것은 아니다.