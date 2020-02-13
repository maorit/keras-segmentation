from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam

from config import LOG_DIR, TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, N_CLASSES, INPUT_WIDTH, INPUT_HEIGHT
from model.fcn import fcn_32
from utils.data_utils import generate_input_data

# 保存的方式，3世代保存一次
checkpoint_period = ModelCheckpoint(
    LOG_DIR + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
    monitor='val_loss',
    save_weights_only=True,
    save_best_only=True,
    period=1
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


def my_loss(y_true, y_pred):
    return K.mean(K.categorical_crossentropy(y_true, y_pred))


if __name__ == '__main__':
    # 获取模型
    model = fcn_32(n_classes=N_CLASSES, input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT)

    # 编译模型
    model.compile(loss=my_loss, optimizer=Adam(), metrics=['accuracy'])

    # 生成训练和验证数据
    train_gen = generate_input_data(stage='train', batch_size=TRAIN_BATCH_SIZE, n_classes=N_CLASSES,
                                    input_width=model.input_width,
                                    input_height=model.input_height, output_width=model.output_width,
                                    output_height=model.output_height)
    val_gen = generate_input_data(stage='val', batch_size=VAL_BATCH_SIZE, n_classes=N_CLASSES,
                                  input_width=model.input_width,
                                  input_height=model.input_height, output_width=model.output_width,
                                  output_height=model.output_height)

    # 训练模型
    model.fit_generator(generator=train_gen, steps_per_epoch=max(1, 200 // TRAIN_BATCH_SIZE), validation_data=val_gen,
                        validation_steps=max(1, 20 // VAL_BATCH_SIZE), epochs=10,
                        callbacks=[checkpoint_period, reduce_lr, early_stopping])

    # # 保存权重
    model.save_weights('./last1.h5')
