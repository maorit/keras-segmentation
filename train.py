import os

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

from config import LOG_DIR, INPUT_HEIGHT, INPUT_WIDTH, TRAIN_BATCH_SIZE, N_CLASSES, VAL_BATCH_SIZE
from losses.focal_loss import focal_loss
from metrics.mean_accuracy import *
from metrics.mean_iou import CategoricalMeanIoU
from model.segnet import segnet
from utils.data_utils import generate_input_data

# 保存的方式，3世代保存一次
checkpoint_period = ModelCheckpoint(
    os.path.join(LOG_DIR + 'ep{epoch:05d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=True,
    period=100
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
    # 获取模型
    model = segnet(21, input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH)
    # 编译模型
    # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4),
    #               metrics=['categorical_accuracy', MeanIoU(num_classes=N_CLASSES),
    #                        acc_of_clazz0, acc_of_clazz1, acc_of_clazz2, acc_of_clazz3, acc_of_clazz4, acc_of_clazz5,
    #                        acc_of_clazz6, acc_of_clazz7, acc_of_clazz8, acc_of_clazz9, acc_of_clazz10, acc_of_clazz11,
    #                        acc_of_clazz12, acc_of_clazz13, acc_of_clazz14, acc_of_clazz15, acc_of_clazz16,
    #                        acc_of_clazz17, acc_of_clazz18, acc_of_clazz19, acc_of_clazz20])
    model.compile(loss=focal_loss(gamma=2, beta=0, positive_only=True), optimizer=Adam(lr=1e-4),
                  metrics=['categorical_accuracy', CategoricalMeanIoU(num_classes=N_CLASSES),
                           acc_of_clazz0, acc_of_clazz1, acc_of_clazz2, acc_of_clazz3, acc_of_clazz4, acc_of_clazz5,
                           acc_of_clazz6, acc_of_clazz7, acc_of_clazz8, acc_of_clazz9, acc_of_clazz10, acc_of_clazz11,
                           acc_of_clazz12, acc_of_clazz13, acc_of_clazz14, acc_of_clazz15, acc_of_clazz16,
                           acc_of_clazz17, acc_of_clazz18, acc_of_clazz19, acc_of_clazz20])

    # 生成训练和验证数据
    train_gen = generate_input_data(stage='train', batch_size=TRAIN_BATCH_SIZE, n_classes=N_CLASSES,
                                    input_width=INPUT_WIDTH,
                                    input_height=INPUT_HEIGHT, output_width=INPUT_WIDTH,
                                    output_height=INPUT_HEIGHT)
    val_gen = generate_input_data(stage='val', batch_size=VAL_BATCH_SIZE, n_classes=N_CLASSES,
                                  input_width=INPUT_WIDTH,
                                  input_height=INPUT_HEIGHT, output_width=INPUT_WIDTH,
                                  output_height=INPUT_HEIGHT)

    # 训练模型
    model.fit_generator(generator=train_gen, steps_per_epoch=max(1, 1500 // TRAIN_BATCH_SIZE), validation_data=val_gen,
                        validation_steps=max(1, 1500 // VAL_BATCH_SIZE), epochs=50,
                        callbacks=[checkpoint_period, reduce_lr, early_stopping])

    # 保存权重,将权重保存到``.last1.h5``文件中
    model.save_weights('.last1.h5')
