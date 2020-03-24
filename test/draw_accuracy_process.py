import pickle

from matplotlib import pyplot as plt

from config import VOC_CLASSES

# 历史记录名
history_filename = r'fcn32_CrossEntropy_null_null_1e4'

# 读取历史记录
with open(f'E:\keras-segmentation\logs\{history_filename}.history', 'rb') as history_file:
    history = pickle.load(history_file)

# 将数据整理为列表
class_acc = [history[f'val_acc_of_clazz{i}'] for i in range(21)] + [history['val_categorical_accuracy']]
class_name = [VOC_CLASSES[i] for i in range(21)] + ['mean_acc']

# 绘图
for class_acc, class_name in zip(class_acc, class_name):
    plt.plot(class_acc)
    plt.legend(class_name)
    plt.show()
plt.title(f'accuracy process|{history_filename}')
plt.show()
