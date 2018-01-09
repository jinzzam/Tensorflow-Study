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

print(len(features))  # python list  500이 나와야함  총 파일 개수가 500개

shuffle_data = True
if shuffle_data:
    c = list(zip(features, labels))
    shuffle(c)
    features, labels = zip(*c)  # zip(*c) :  unzip 하기

features = np.array(features)
labels = np.array(labels)

print(len(labels))  # np array
print(labels)  # features, labels는 python tuple이다. np array로 반환해준다.
print(labels.shape)

image = features[0]  # 첫번째 이미지 파일 선택
image = image.reshape((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL))  # 역변환
image = image.astype(np.uint8)  # data type 을 위 코드에서 float32 로 변경했기 때문에 출력하기 전에 다시 원래 타입으로 변경해야함
cv2.imshow('Restored Image', image)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
# 위 코드 : Testing

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
    print(images_batch_val)
    print(labels_batch_val)

# placeholder를 통해 네트워크에 이미지와 라벨 공급
images_batch = tf.placeholder(dtype=tf.float32, shape=[None, IMG_HEIGHT * IMG_WIDTH * NUM_CHANNEL])
# 입력 이미지가 제공될 placeholder의 크기는 batch_size*image_size 이다.
# None은 나중에 실제로 제공되는 데이터의 크기에 맞춘다는 의미
labels_batch = tf.placeholder(dtype=tf.int32, shape=[None, ])
# 크기가 미정인 1차원 벡터의 shape은 이렇게 지정함 (이렇게하면 label이 one hot encoding이 아님)

w1 = tf.get_variable("w1", [IMG_HEIGHT * IMG_WIDTH * NUM_CHANNEL, 1024])
# weight 행렬의 크기는 '이전 레이어의 노드 개수 * 현재 레이어의 노드 개수'
b1 = tf.get_variable("b1", [1024])

# Layer1의 출력 벡터
fc1 = tf.nn.relu(tf.matmul(images_batch, w1) + b1)
# images_batch * w1은 [batch_size, 1024]크기의 행렬이고 b1은 [1024]크기의 1차원 벡터이다.
# 행렬의 각 행에 b1이 더해진다. 이것을 numpy에서는 broadcasting이라고 부른다.

w2 = tf.get_variable("w2", [1024, 512])
b2 = tf.get_variable("b2", [512])

# 이전 레이어의 출력(fc1)이 이 레이어의 입력이 된다.
fc2 = tf.nn.relu(tf.matmul(fc1, w2) + b2)

w3 = tf.get_variable("w3", [512, NUM_CLASS])
b3 = tf.get_variable("b3", [NUM_CLASS])
y_pred = tf.matmul(fc2, w3) + b3
# 출력 레이어에는 아직 activation 함수를 적용하지 않았다.
# y_pred는 임의의 실수를 원소로 가지는 batch_size * NUM_CLASS 크기의 행렬이다.

# y_pred는 아직 softmax 함수를 적용하기 전이고 labels_batch는 one hot encoding이 아니다.
# 이런 상황에서 필요한 모든 일을 대신 처리해주는 함수가 아래의 함수.
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=labels_batch)
loss_mean = tf.reduce_mean(loss)
# 평균 loss 계산

# 지금까지 정의한 모든 노드들이 결국 어떤 값을 표현한다면 train_op 는 알고리즘(operation)을 표현한다.
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 지금까지 구성한 그래프를 실행할 session을 생성하고 변수들을 초기화한다.

iter_ = train_data_iterator()
for step in range(500):
    images_batch_val, labels_batch_val = next(iter_)
    _, loss_val = sess.run([train_op, loss_mean],       # 이렇게 실행할 타겟을 지정
                           feed_dict={                      # Python dictionary의 형태로 placeholder에 제공할 데이터 공급
                               images_batch:images_batch_val,
                               labels_batch:labels_batch_val
                           })
    print(loss_val)

print('Training Finished....')

print('Test begins...')
TEST_BSIZE = 50
for i in range(int(len(test_features)/TEST_BSIZE)):
    images_batch_val = test_features[i*TEST_BSIZE:(i+1)*TEST_BSIZE] / 255.
    labels_batch_val = test_labels[i*TEST_BSIZE:(i+1)*TEST_BSIZE]

    loss_val = sess.run([loss_mean], feed_dict={
        images_batch:images_batch_val,
        labels_batch:labels_batch_val
    })
    print(loss_val)
