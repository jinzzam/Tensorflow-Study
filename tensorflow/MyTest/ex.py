import tensorflow as tf

a = tf.constant(5, name='input_a')
b = tf.constant(7, name='input_b')

c = tf.multiply(a, b, name='mul_c')
d = tf.add(a, b, name='add_d')
e = tf.add(c, d, name='add_e')

sess = tf.Session()  # 세션을 만든다
print(sess.run(e))

# # 상수의 선언
# def constant(value, dtype=None, name="Const", verify_shape=False)
# # 상수 하나를 선언한다
#
# # 명령(op)의 선언
# def multiply(x, y, name=None)
# def add(x, y, name=None)

writer = tf.summary.FileWriter('./ex', sess.graph)