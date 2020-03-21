import numpy as np

from config import N_CLASSES, INPUT_WIDTH, INPUT_HEIGHT, PALETTE
from model.fcn import fcn_8
from utils.data_utils import _load_image

# 获取模型
model = fcn_8(n_classes=N_CLASSES, input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT)

# 加载权重
# model.load_weights(r"C:\Users\chenhai\Downloads\last1.h5")

# 进行预测
output_height = INPUT_HEIGHT
output_width = INPUT_WIDTH
# r'data/VOCdevkit/VOC2007/SegmentationClass/000063.png'
test_data = _load_image(image_path=r'data\VOCdevkit\VOC2012\JPEGImages\2007_000033.jpg',
                        width=INPUT_WIDTH, height=INPUT_HEIGHT)
test_data = np.array([test_data])
output = model.predict(test_data)
output = output[0]
output = output.argmax(axis=1)
output.resize(output_width, output_height)
output_img = np.zeros((output_width, output_height, 3))
for c in range(N_CLASSES):
    output_img[:, :, 0] += (output == c).astype(np.int) * PALETTE[c, 0]
    output_img[:, :, 1] += (output == c).astype(np.int) * PALETTE[c, 1]
    output_img[:, :, 2] += (output == c).astype(np.int) * PALETTE[c, 2]

import matplotlib.pyplot as plt

plt.imshow(output_img)
plt.show()
