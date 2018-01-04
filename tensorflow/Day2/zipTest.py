x = [1, 2, 3]
y = [4, 5, 6]
zipped = zip(x, y)
print(zipped)

zipped_list = list(zipped)
print(zipped_list)
x2, y2 = zip(*zipped_list)
print(x == list(x2) and y == list(y2))