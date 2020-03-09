from collections import Counter

import numpy as np
from PIL import Image

idx_path = r'../data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'

with open(idx_path, 'r') as f:
    idxs = f.readlines()

img_path = r'../data/VOCdevkit/VOC2012/SegmentationClass/'

sum = Counter()

from tqdm import tqdm

for idx in tqdm(idxs):
    idx = idx.strip()
    img = Image.open('{}{}.png'.format(img_path, idx))
    img = np.array(img)
    # print(Counter(img.flatten()))
    sum.update(Counter(img.flatten()))

print(sum)

# 得到结果
Counter({0: 182014429, 255: 14335564, 15: 11995853, 8: 6752515, 6: 4375622, 12: 4344951, 19: 3984238, 18: 3612229, 7: 3494749, 11: 3381632, 14: 2888641, 9: 2861091, 20: 2349235, 13: 2283739, 17: 2254463, 3: 2232247, 10: 2060925, 1: 1780580, 16: 1670340, 5: 1517186, 4: 1514260, 2: 758311})