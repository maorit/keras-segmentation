from PIL import Image

idx_path = r'../data/VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt'

with open(idx_path, 'r') as f:
    idxs = f.readlines()

img_path = r'../data/VOCdevkit/VOC2007/JPEGImages/'

height = 1000
width = 1000

for idx in idxs:
    idx = idx.strip()
    img = Image.open('{}{}.jpg'.format(img_path, idx))

    height = min(height, img.height)
    width = min(width, img.width)

print('height={},width={}'.format(height, width))
# height=240,width=292