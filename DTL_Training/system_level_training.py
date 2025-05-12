import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# 确保路径正确
model_path = r'E:\temp\oscillation_detect\DTL-Location-main\DTL-Location-main\DTL FO_Location\DTL_Training\system_level_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"File not found: {model_path}")

# 加载系统级别的预训练模型
base_model = load_model(model_path)
base_model.trainable = False  # 冻结预训练模型的卷积层

# 设置全局图片路径
global_output_dir = '../MoreSamples/global_output'

# 加载数据
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.15)  # 数据归一化并划分验证集
train_generator = datagen.flow_from_directory(
    global_output_dir,
    target_size=(224, 224),  # VGG16 的输入尺寸
    batch_size=16,
    class_mode='categorical',
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    global_output_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# 构建自定义分类层
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(train_generator.num_classes, activation='softmax')  # 分类数根据数据集自动调整
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 添加 Early Stopping 回调
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # 监控验证集准确率
    patience=5,              # 如果 5 个 epoch 内验证集准确率没有提升，则停止训练
    restore_best_weights=True  # 恢复验证集表现最好的权重
)

# 训练模型
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[early_stopping],  # 添加 Early Stopping 回调
    verbose=1
)

# 保存模型
model.save('system_level_model.h5')

# 打印训练结果
print("System-level model training complete.")

# 可视化训练过程
def plot_training_history(history):
    # 绘制训练和验证的准确率曲线
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制训练和验证的损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制学习率曲线（如果有动态学习率）
    if 'lr' in history.history:
        plt.subplot(2, 2, 3)
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()

    # 显示图像
    plt.tight_layout()
    plt.show()

# 绘制混淆矩阵
def plot_confusion_matrix(model, validation_generator):
    # 获取真实标签和预测标签
    y_true = validation_generator.classes
    y_pred = np.argmax(model.predict(validation_generator), axis=1)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    labels = list(validation_generator.class_indices.keys())

    # 绘制混淆矩阵
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.show()

# 调用可视化函数
plot_training_history(history)
plot_confusion_matrix(model, validation_generator)