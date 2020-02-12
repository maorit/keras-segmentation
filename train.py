import random

from PIL import Image
from keras import Sequential
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

from config import *


def get_crop_offset(img):
    left = random.randint(0, img.width - TARGET_WIDTH)
    upper = random.randint(0, img.height - TARGET_HEIGHT)
    return (left, upper)


def generate_data_from_file(idx_path, batch_size):
    # 读入图片的编号
    with open(idx_path, 'r') as f:
        idxs = f.readlines()

    i = 0
    while True:
        imgs = []
        masks = []
        for _ in range(batch_size):
            # 读入样本的编号
            idx = idxs[i].strip()

            # 读入原图像
            img = Image.open('{}{}.jpg'.format(img_dir, idx))
            (left, upper) = get_crop_offset(img)
            img = np.array(img)
            img = img[upper:upper + TARGET_HEIGHT, left:left + TARGET_WIDTH, :]
            img = img / 255
            imgs.append(img)

            # 读入mask
            mask_img = Image.open('{}{}.png'.format(mask_dir, idx)).convert('RGB')
            mask_img = np.array(mask_img)
            mask_img = mask_img[upper:upper + TARGET_HEIGHT, left:left + TARGET_WIDTH, :]
            mask = np.zeros((mask_img.shape[0], mask_img.shape[1], NCLASSES))
            for c in range(NCLASSES):
                mask[:, :, c] = (mask_img == (palette[c].reshape(1, 1, 3))).all(axis=2)
            mask = mask.reshape(-1, NCLASSES)
            masks.append(mask)

            # 断言
            assert img.shape[0:2] == mask_img.shape[0:2]

            # 转向下一个样本
            i = (i + 1) % len(idxs)

        yield (np.array(imgs), np.array(masks))


def loss(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.sum(cross_entropy) / (TARGET_HEIGHT * TARGET_WIDTH)
    return cross_entropy


def get_model(model_name='default'):
    return Sequential()


# 保存的方式，3世代保存一次
checkpoint_period = ModelCheckpoint(
    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=True,
    period=3
)
# 学习率下降的方式，val_loss3次不下降就下降学习率继续训练
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=1
)
# 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=10,
    verbose=1
)

if __name__ == '__main__':
    model = get_model()
    model.compile(loss=loss, optimizer=Adam(), metrics=['accuracy'])
    model.fit_generator(generator=generate_data_from_file(idx_path=train_idx_path, batch_size=batch_size),
                        steps_per_epoch=max(1, 200 // batch_size),
                        validation_data=generate_data_from_file(idx_path=val_idx_path, batch_size=batch_size),
                        epochs=10,
                        callbacks=[checkpoint_period, reduce_lr, early_stopping])
    model.save_weights(log_dir + 'last1.h5')
