from config import *

from model.fcn import fcn_32
from utils.data_utils import _get_image_array

# 获取模型
model = fcn_32(n_classes=N_CLASSES, input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT)

# 加载权重
model.load_weights("last1.h5_2020_2_13.bak")

# 进行预测
output_height = model.output_height
output_width = model.output_width
# r'data/VOCdevkit/VOC2007/SegmentationClass/000063.png'
test_data = _get_image_array(image_path=Path(r'E:\keras-segmentation\data\VOCdevkit\VOC2007\JPEGImages\000738.jpg'),
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
