import numpy as np
import os
import cv2
from random import shuffle

# 이미지 크기는 60*60으로 resize해서 사용할 계획
IMG_HEIGHT = 60
IMG_WIDTH = 60
NUM_CHANNEL = 3
NUM_CLASS = 5   # 분류할 동물의 종류 가지 수
BATCH_SIZE = 50

def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)    # 데이터 타입 변환

    return img

IMAGE_DIR_BASE = r'C:\Users\jin\Documents\Tensorflow-Study\tensorflow\animal_images\animal_images'
image_dir_list = os.listdir(IMAGE_DIR_BASE)     # 하위 파일 이름 목록       os.listdir 함수를 이용하여 디렉토리에 속한 파일과 서브 디렉토리의 목록을 읽는다
print(image_dir_list)       # 동물 종류 폴더 이름 나열하여 출력해줌

class_index = 0

features = []       # 데이터       이미지가 저장된 순서대로 저장됨
labels = []         # 정답 (분류)   0(cat), 1(cow) 2(dog) 3(pig) 4(sheep)

for dir_name in image_dir_list:
    image_file_list = os.listdir(IMAGE_DIR_BASE + os.sep + dir_name)    # 경로명 + / + 파일이름
    for file_name in image_file_list:
        image = load_image(IMAGE_DIR_BASE + os.sep + dir_name + os.sep + file_name)

        features.append(image.ravel())      # ravel()은 다차원배열을 1차원 배열로 반환함  numpy array가 제공해주는 함수임
        labels.append(class_index)          # append는 python리스트에 새로운 값을 추가함

    class_index += 1    # 디렉토리가 바뀔 때 label 값을 1 증가시킴

print(len(features))        # python list  500이 나와야함  총 파일 개수가 500개

shuffle_data = True
if shuffle_data:
    c = list(zip(features, labels))
    shuffle(c)
    features, labels = zip(*c)          # zip(*c) :  unzip 하기

features = np.array(features)
labels = np.array(labels)

print(len(labels))      # np array
print(labels)           # features, labels는 python tuple이다. np array로 반환해준다.
print(labels.shape)

image = features[0]     # 첫번째 이미지 파일 선택
image = image.reshape((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL))     # 역변환
image = image.astype(np.uint8)      # data type 을 위 코드에서 float32 로 변경했기 때문에 출력하기 전에 다시 원래 타입으로 변경해야함
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

def train_data_iterator():      # generator?
    batch_idx = 0
    while True:
        # 매 시대마다 새로 shuffling 한다
        idxs = np.arange(0, len(train_features))
        # np.arange(A, B) : 배열에 A이상 B미만 까지의 숫자를 순서대로 저장해줌
        np.random.shuffle(idxs)     # 인덱스 자체를 셔플
        shuf_features = train_features[idxs]
        shuf_labels = train_labels[idxs]

        batch_size = BATCH_SIZE     # 50

        for batch_idx in range(0, len(train_features), batch_size):     # array slicing
            images_batch = shuf_features[batch_idx:batch_idx + batch_size] / 255.   # 255로 나누어서 0에서 1사이의 실수로 정규화한다
            labels_batch = shuf_labels[batch_idx:batch_idx + batch_size]
            yield images_batch, labels_batch

iter_ = train_data_iterator()       # generator를 호출하면 iterator를 반환한다
for step in range(100):
    # get a batch of data
    images_batch_val, labels_batch_val = next(iter_)       # iterator에 대해서 next함수를 실행할 때마다 하나의 배치가 반환된다
    print(images_batch_val)
    print(labels_batch_val)