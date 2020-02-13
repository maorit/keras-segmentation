from PIL import Image
from keras import Sequential

from config import *


def _get_model(model_name='default'):
    return Sequential()


model = _get_model()
model.load_weights("logs/ep021-loss0.083-val_loss0.143.h5")

img = Image.open('{}{}.jpg'.format(IMG_DIR, '000033'))
y_pred = model.predict(img)  # (N, 21)
