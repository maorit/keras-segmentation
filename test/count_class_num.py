import os
from collections import Counter

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from config import VOC_CLASSES, INPUT_WIDTH, INPUT_HEIGHT

# 获取训练集图片id
with open('../data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', 'r') as f:
    idxs = f.readlines()

# 统计量
sum = Counter()  # 像素点总数
appear_time = Counter()  # 实例出现的图片数

# 遍历图片进行统计
for idx in tqdm(idxs):
    img_data = Image.open(os.path.join('../data/VOCdevkit/VOC2012/SegmentationClass/', f'{idx.strip()}.png')).resize(
        (INPUT_WIDTH, INPUT_HEIGHT), Image.NEAREST)
    img_data = np.array(img_data)
    sum.update(Counter(img_data.flatten()))
    appear_time.update(Counter(np.unique(img_data)))

print(sum)
print(appear_time)
# sum = Counter({0: 266362305, 255: 21047207, 15: 17556354, 8: 9831255, 6: 6484471, 12: 6181636, 19: 5824265, 18: 5296032,7: 5227368, 11: 4778217, 9: 4209604, 14: 4162999, 13: 3404100, 20: 3306567, 17: 3246995, 3: 3157189,10: 3080422, 1: 2703473, 16: 2373880, 4: 2250702, 5: 2211011, 2: 1082764})
# appear_time = Counter({255: 1464, 0: 1460, 15: 442, 9: 148, 8: 131, 7: 128, 12: 121, 3: 105, 18: 93, 1: 88, 5: 87, 20: 83, 19: 83, 11: 82,16: 82, 14: 81, 4: 78, 6: 78, 13: 68, 2: 65, 10: 64, 17: 63})
# class_freq = {0: 0.6959531209240221, 1: 0.11719239841807973, 2: 0.06354487492487981, 3: 0.1147020975748698, 4: 0.11007367647611178, 5: 0.0969464093789287, 6: 0.31713197170159757, 7: 0.15578770637512207, 8: 0.28628443943635196, 9: 0.10850246532543285, 10: 0.18360745906829834, 11: 0.22228599176174257, 12: 0.1948848755891658, 13: 0.1909648670869715, 14: 0.19605655434690875, 15: 0.1515207592718202, 16: 0.11043455542587652, 17: 0.1966079833015563, 18: 0.2172339654737903, 19: 0.2676844309611493, 20: 0.151970507150673}

# 绘制各个类别像素点的频率图
class_freq = [(sum[i] / (appear_time[i] * INPUT_WIDTH * INPUT_HEIGHT)) for i in range(21)]
class_name = [VOC_CLASSES[i] for i in range(21)]
plt.barh(y=class_name, width=class_freq)
plt.title('the frequency of different class in Pascal VOC')
plt.xlabel('frequency')
plt.ylabel('class_name')
plt.box(False)
plt.show()
