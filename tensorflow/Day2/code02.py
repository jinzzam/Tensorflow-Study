import numpy as np
import os
import cv2
from random import  shuffle


IMG_HEIGHT = 60
IMG_WIDTH = 60
NUM_CHANNEL = 3
NUM_CLASS = 5
BATCH_SIZE = 50

def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)

    return img

IMAGE_DIR_BASE = r'C:\Users\jin\Documents\Tensorflow-Study\tensorflow\animal_images\animal_images'
image_dir_list = os.listdir(IMAGE_DIR_BASE)
print(image_dir_list)

class_index = 0

features = []       # 데이터
labels = []         # 정답 (분류)

for dir_name in image_dir_list:
    image_file_list = os.listdir(IMAGE_DIR_BASE + os.sep + dir_name)    # 경로명 + / + 파일이름
    for file_name in image_file_list:
        image = load_image(IMAGE_DIR_BASE + os.sep + file_name)

        features.append(image.ravel())      # ravel()은 다차원배열을 1차원 배열로 반환함  numpy array가 제공해주는 함수임
        labels.append(class_index)          # append는 python리스트에 새로운 값을 추가함

    class_index += 1

print(len(features))        # python list  500이 나와야함  총 파일 개수가 500개

shuffle_data = True
if shuffle_data:
    c = list(zip(features, labels))
    shuffle(c)
    features, labels = zip(*c)          # * c =  unzip 하기

features = np.array(features)
labels = np.array(labels)

print(len(labels))      # np array
print(labels)           # features, labels는 python tuple이다. np array로 반환해준다.
print(labels.shape)

image = features[0]
image = image.reshape((IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL))     # 역변환
image = image.astype(np.uint8)      # data type 을 위 코드에서 float32 로 변경했기 때문에 출력하기 전에 다시 원래 타입으로 변경해야함
cv2.imshow('Restored Image', image)
cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()
# 위 코드 : Testing

train_features = features[0:int(0.8 * len(features))]
train_labels = labels[0:int(0.8 * len(labels))]

# val_features = features[int(0.6 * len(features)):int(0.8 * len(features))]
# val_labels = labels[int(0.6 * len(features)):int(0.8 * len(features))]

test_features = features[int(0.8 * len(features)):]
test_labels = labels[int(0.8 * len(labels)):]

def train_data_iterator():
    batch_idx = 0
    while True:
        idxs = np.arrange(0, len(train_features))
        np.random.shuffle(idxs)
        shuf_features = train_features[idxs]
        shuf_labels = train_labels[idxs]

        batch_size = BATCH_SIZE

        for batch_idx in range(0, len(train_features), batch_size):     # array slicing
            images_batch = shuf_features[batch_idx:batch_idx + batch_size] / 255.
            labels_batch = shuf_labels[batch_idx:batch_idx + batch_size]
            yield images_batch, labels_batch

iter_ = train_data_iterator()
for step in range(100):
    # get a batch of data
    images_batch_val, labels_batch_val = next(iter_)
    print(images_batch_val)
    print(labels_batch_val)