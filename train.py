import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

# 保存的方式，3世代保存一次
from config import LOG_DIR, INPUT_HEIGHT, INPUT_WIDTH, TRAIN_BATCH_SIZE, N_CLASSES, VAL_BATCH_SIZE
from losses.focal_loss import focal_loss
from metrics.mean_accuracy import mean_acc
from metrics.mean_iou import mean_iou
from model.fcn import fcn_8
from utils.data_utils import generate_input_data

checkpoint_period = ModelCheckpoint(
    LOG_DIR + 'ep{epoch:05d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
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
    model = fcn_8(21, input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH)
    # 编译模型
    model.compile(loss=focal_loss(alpha=np.ones(21)), optimizer=Adam(lr=1e-4),
                  metrics=['categorical_accuracy', mean_iou(21), mean_acc(21)])

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
