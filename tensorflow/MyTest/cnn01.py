import os
from random import shuffle

import cv2
import numpy as np
import tensorflow as tf

# 이미지 크기는 60*60으로 resize해서 사용할 계획
IMG_HEIGHT = 60
IMG_WIDTH = 60
NUM_CHANNEL = 3
NUM_CLASS = 5  # 분류할 동물의 종류 가지 수
BATCH_SIZE = 50


def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)  # 데이터 타입 변환

    return img


IMAGE_DIR_BASE = r'C:\Users\jin\Documents\Tensorflow-Study\tensorflow\animal_images\animal_images'
image_dir_list = os.listdir(IMAGE_DIR_BASE)  # 하위 파일 이름 목록       os.listdir 함수를 이용하여 디렉토리에 속한 파일과 서브 디렉토리의 목록을 읽는다
print(image_dir_list)  # 동물 종류 폴더 이름 나열하여 출력해줌

class_index = 0

features = []  # 데이터       이미지가 저장된 순서대로 저장됨
labels = []  # 정답 (분류)   0(cat), 1(cow) 2(dog) 3(pig) 4(sheep)

for dir_name in image_dir_list:
    image_file_list = os.listdir(IMAGE_DIR_BASE + os.sep + dir_name)  # 경로명 + / + 파일이름
    for file_name in image_file_list:
        image = load_image(IMAGE_DIR_BASE + os.sep + dir_name + os.sep + file_name)

        features.append(image.ravel())  # ravel()은 다차원배열을 1차원 배열로 반환함  numpy array가 제공해주는 함수임
        labels.append(class_index)  # append는 python리스트에 새로운 값을 추가함

    class_index += 1  # 디렉토리가 바뀔 때 label 값을 1 증가시킴

shuffle_data = True
if shuffle_data:
    c = list(zip(features, labels))
    shuffle(c)
    features, labels = zip(*c)  # zip(*c) :  unzip 하기

features = np.array(features)
labels = np.array(labels)

# training set 8:2 비율이 일반적으로 사용된다
train_features = features[0:int(0.8 * len(features))]
train_labels = labels[0:int(0.8 * len(labels))]

# Validation data set 언제까지 training 을 지속할지 결정하기 위해서 사용된다
# 보통 6:2:2의 비율로 분할한다  지금은 사용하지 않는다
# val_features = features[int(0.6 * len(features)):int(0.8 * len(features))]
# val_labels = labels[int(0.6 * len(features)):int(0.8 * len(features))]

# test set
test_features = features[int(0.8 * len(features)):]
test_labels = labels[int(0.8 * len(labels)):]


def train_data_iterator():  # generator?
    batch_idx = 0
    while True:
        # 매 시대마다 새로 shuffling 한다
        idxs = np.arange(0, len(train_features))
        # np.arange(A, B) : 배열에 A이상 B미만 까지의 숫자를 순서대로 저장해줌
        np.random.shuffle(idxs)  # 인덱스 자체를 셔플
        shuf_features = train_features[idxs]
        shuf_labels = train_labels[idxs]

        batch_size = BATCH_SIZE  # 50

        for batch_idx in range(0, len(train_features), batch_size):  # array slicing
            images_batch = shuf_features[batch_idx:batch_idx + batch_size] / 255.  # 255로 나누어서 0에서 1사이의 실수로 정규화한다
            labels_batch = shuf_labels[batch_idx:batch_idx + batch_size]
            yield images_batch, labels_batch


iter_ = train_data_iterator()  # generator를 호출하면 iterator를 반환한다
for step in range(100):
    # get a batch of data
    images_batch_val, labels_batch_val = next(iter_)  # iterator에 대해서 next함수를 실행할 때마다 하나의 배치가 반환된다
#    print(images_batch_val)
#    print(labels_batch_val)


# === Convolution Neural Network ===

images_batch = tf.placeholder(dtype=tf.float32, shape=[None, IMG_HEIGHT*IMG_WIDTH*NUM_CHANNEL])
x_image = tf.reshape(images_batch, [-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL])
# -1은 None과 같은 의미
# load image함수에서 이미지를 flatten해두는 바람에 여기에서 다시 3차원 배열로 복구해야 한다.

labels_batch = tf.placeholder(dtype=tf.int32, shape=[None, ])

# ===First Convolution + Pooling Layer===

w_conv1 = tf.get_variable(name='w_conv1', shape=[5, 5, 3, 32], dtype=tf.float32)
# 5X5 크기의 3채널 kernel 32개를 사용

b_conv1 = tf.get_variable(name='b_conv1', shape=[32], dtype=tf.float32)

h_conv1 = tf.nn.relu(
    tf.nn.conv2d(x_image, w_conv1, strides=[1, 1, 1, 1], padding='SAME')
    +b_conv1
                       # [batch_size, image_height, image_width, num_channel
)

h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# ===Second Concolution + Pooling Layer===

w_conv2 = tf.get_variable(name='w_conv2', shape=[5, 5, 32, 64], dtype=tf.float32)
# 5X5 크기의 3채널 kernel 32개를 사용

b_conv2 = tf.get_variable(name='b_conv2', shape=[64], dtype=tf.float32)

h_conv2 = tf.nn.relu(
    tf.nn.conv2d(h_pool1, w_conv2, strides=[1, 1, 1, 1], padding='SAME')
    +b_conv2
                       # [batch_size, image_height, image_width, num_channel
)

h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# ===Fully Connected Layer===

w1 = tf.get_variable("w1", [IMG_HEIGHT//4*IMG_WIDTH//4*64, 1024])
# max pooling 을 2번 지나면서 이미지 크기가 1/4로 줄었고
# 마지막 convolution layer에서 64개의 kernel을 사용했으므로 총 값의 개수는 이렇게 된다.
b1 = tf.get_variable("b1", [1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, IMG_HEIGHT//4*IMG_WIDTH//4*64])
# FC층에 입력하기 위해서 flatten 해 줌

fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w1)+b1)

# Dropout Layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(fc1, keep_prob)


# Output Layer
w2 = tf.get_variable("w2", [1024, NUM_CLASS])
b2 = tf.get_variable("b2", [NUM_CLASS])

y_pred = tf.matmul(h_fc1_drop, w2) + b2

class_prediction = tf.arg_max(y_pred, 1, output_type=tf.int32)

correct_prediction = tf.equal(class_prediction, labels_batch)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Loss and Train_op
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss_mean)

# Training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

iter_ = train_data_iterator()
for step in range(500):
    images_batch_val, labels_batch_val = next(iter_)
    accuracy_, _, loss_val = sess.run([accuracy, train_op, loss_mean],
                                      feed_dict={
                                          images_batch: images_batch_val,
                                          labels_batch: labels_batch_val,
                                          keep_prob: 0.5
                                      })
    print('Iteration {}: {}, {}'.format(step, accuracy_, loss_val))

print('Training Finished....')

# Testing
print('Test begins....')

TEST_BSIZE = 50
for i in range(int(len(test_features)/TEST_BSIZE)):
    images_batch_val = test_features[i*TEST_BSIZE:(i+1)*TEST_BSIZE]/255.
    labels_batch_val = test_labels[i*TEST_BSIZE:(i+1)*TEST_BSIZE]

    loss_val, accuracy_ = sess.run([loss_mean, accuracy],
                                   feed_dict={
                                       images_batch: images_batch_val,
                                       labels_batch: labels_batch_val,
                                       keep_prob: 1.0
    })
    print('ACC = {}, LOSS = {}'.format(accuracy_, loss_val))