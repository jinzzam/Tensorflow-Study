import cv2
from matplotlib import pyplot as plt

image_path = r'C:\Users\jin\Documents\Tensorflow-Study\tensorflow\animal_images\animal_images\cat\images-22.jpeg'

img = cv2.imread(image_path)        # img 는 numpy array이다. 행렬이나 벡터 연산을 하기 위한 추가적인 라이브러리가 제공함.
# imread : 파일을 비트맵으로 변환시켜 반환함

print(img.ndim)     # 배열의 차원 - 3차원 배열임 : RGB 세가지 속성을 가지고 있기 때문
print(img.shape)    #
print(img.dtype)    # 각 원소의 데이터 타입을 알려줌 unsigned integer 8 bit = 0~255까지의 정수값 표현
print(len(img))

print(img)
print(img.tolist())

cv2.imshow('Test Image', img)
cv2.waitKey(0) & 0xFF       # 아무 키를 받을 때까지 기다려라
cv2.destroyAllWindows()     # 창 종료

img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# opencv는 기본적으로 이미지파일을 BGR로 저장함
# 그래서 저장하기 전에 RGB로 바꿔줌

cv2.imwrite('converted.jpg', img)

# matplotlib를 이용한 파일 보기
plt.imshow(img)
plt.show()

def plot_images(image):
    fig, axes = plt.subplots(4, 4)
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

    for i, ax in enumerate(axes.flat) :
        row = i // 4        # 몫을 구하는 연산 //
        col = i % 4
        image_frag = image[row*50:(row+1)*50, col*50:(col+1)*50, :]  # 세로, 가로, 처음부터끝까지(z축)  그래서 가로세로 4등분
        ax.imshow(image_frag)

        xlabel = '{},{}'.format(row, col)
        ax.set_xlabel(xlabel)   # x축 제목?
        ax.set_xticks([])  # Remove ticks from the plot.    # x축 눈금 변경 (공백 -> remove)
        ax.set_yticks([])                                   # y축 눈금 변경 (공백 -> remove)
    plt.show()

plot_images(img)