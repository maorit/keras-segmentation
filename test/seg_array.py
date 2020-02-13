from pathlib import Path

import cv2
import numpy as np

from config import PALETTE, SEG_DIR

height = 416
width = 610
n_classes = 21
id = '000904'
seg_image_path = Path.joinpath(SEG_DIR, f'{id}.png')

seg_img = cv2.imread(str(seg_image_path.absolute()), cv2.IMREAD_COLOR)
seg_img = cv2.resize(seg_img, (width, height), interpolation=cv2.INTER_NEAREST)
seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

seg_labels = np.zeros((height, width, n_classes))

pers = 0
# 验证其他颜色
for c in range(n_classes):
    seg_labels[:, :, c] = (seg_img == PALETTE[c].reshape(1, 1, 3)).all(axis=2).astype(np.float)
    per = (seg_img == PALETTE[c].reshape(1, 1, 3)).all(axis=2).sum()
    pers += per / (height * width)
    print('{}:{}'.format(c, per / (height * width)))

# 验证背景
per = (seg_img == np.array([224, 224, 192]).reshape(1, 1, 3)).all(axis=2).sum()
pers += per / (height * width)
print('{}:{}'.format(c, per / (height * width)))

print('pers=' + str(pers))
