import  numpy as np
import  os
import  cv2
import glob

IMG_HEIGHT = 60
IMG_WIDTH = 60
NUM_CHANNEL = 3
NUM_CLASS = 5

def load_image(addr) :
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
        labels.append(class_index)          #append는 python리스트에 새로운 값을 추가함

    class_index += 1

print(len(features))        #python list