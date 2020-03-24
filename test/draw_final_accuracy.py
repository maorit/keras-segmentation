import pickle

from matplotlib import pyplot as plt

from config import VOC_CLASSES

# 历史记录名
history_filename = r'fcn32_CrossEntropy_null_null_1e4'

# 读取历史记录
with open(f'E:\keras-segmentation\logs\{history_filename}.history', 'rb') as history_file:
    history = pickle.load(history_file)

# 将数据整理为列表
class_acc = [history[f'val_acc_of_clazz{i}'][-1] for i in range(21)] + [history['val_categorical_accuracy'][-1]]
class_name = [VOC_CLASSES[i] for i in range(21)] + ['mean_acc']

# 绘图
rects = plt.barh(y=class_name, width=class_acc)
for rect in rects:
    plt.text(1, rect.get_y() + 0.5 * rect.get_height(), f'{rect.get_width() * 100:.02f}%')
rects[-1].set_color('orange')
plt.box(False)
plt.title(f'final accuracy|{history_filename}')
plt.show()
