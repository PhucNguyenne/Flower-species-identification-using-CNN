import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import AdamW

# Thiết lập tham số
im_height, im_width, batch_size, epochs = 224, 224, 64, 20
data_path = 'jpeg-224x224/'
os.makedirs(data_path, exist_ok=True)

# Data augmentation cho train - Thêm brightness_range
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.5, 1.5],  # Thêm điều chỉnh độ sáng (0.5 = tối hơn, 1.5 = sáng hơn)
    fill_mode='nearest'
)

# Chỉ rescale cho validation
val_image_generator = ImageDataGenerator(rescale=1./255)

# Thu thập dữ liệu train
train_base_path = 'jpeg-224x224/train/'
all_train_images = []
image_extensions = ['.jpg', '.png', '.jpeg']
for dp, dn, filenames in os.walk(train_base_path):
    label = os.path.basename(dp)
    all_train_images.extend([(os.path.join(dp, f), label) for f in filenames
                            if os.path.splitext(f)[1].lower() in image_extensions])

if not all_train_images:
    print("Không tìm thấy ảnh nào trong thư mục train!")
    exit()
print(f"Tổng số ảnh train: {len(all_train_images)}")

train_df = pd.DataFrame({'filename': [img[0] for img in all_train_images],
                        'class': [img[1] for img in all_train_images]})

# Tạo generator cho train
train_data_gen = train_image_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='class',
    batch_size=batch_size,
    shuffle=True,
    target_size=(im_height, im_width),
    class_mode='categorical'
)

# Tạo generator cho validation
val_dir = os.path.join(data_path, "val/")
if not os.path.exists(val_dir):
    print("Thư mục val/ không tồn tại! Vui lòng tạo thư mục và thêm dữ liệu.")
    exit()

val_data_gen = val_image_generator.flow_from_directory(
    directory=val_dir,
    batch_size=batch_size,
    shuffle=False,
    target_size=(im_height, im_width),
    class_mode='categorical'
)

# Tạo mô hình DenseNet121
covn_base = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False,
                                            input_shape=(224, 224, 3))
covn_base.trainable = True
for layer in covn_base.layers[:-150]:
    layer.trainable = False

model = Sequential([
    covn_base,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.02)),  # Thêm lớp Dense mới
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer=l2(0.02)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(train_data_gen.class_indices), activation='softmax',
                kernel_regularizer=l2(0.02))
])

# In tóm tắt mô hình
model.summary()

# Compile mô hình
model.compile(
    optimizer=AdamW(learning_rate=0.0005, weight_decay=0.01, clipnorm=1.0),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
    ModelCheckpoint(filepath='best_model.keras', monitor='val_loss',
                   save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

# Tính steps
steps_per_epoch = train_data_gen.n // batch_size
validation_steps = val_data_gen.n // batch_size

print(f"Số mẫu train: {train_data_gen.n}")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Số mẫu validation: {val_data_gen.n}")
print(f"Validation steps: {validation_steps}")

# Huấn luyện với dataset lặp lại
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_data_gen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, im_height, im_width, 3], [None, len(train_data_gen.class_indices)])
).repeat()

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_data_gen,
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, im_height, im_width, 3], [None, len(val_data_gen.class_indices)])
).repeat()

history = model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_dataset,
    validation_steps=validation_steps,
    callbacks=callbacks
)

# Vẽ biểu đồ train và validation
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label='Train Loss')
plt.plot(history.history["val_loss"], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label='Train Accuracy')
plt.plot(history.history["val_accuracy"], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('train_val_metrics.png')
plt.close()

# Dự đoán trên ảnh test
test_images = [
    "/content//jpeg-224x224/test/003882deb.jpeg",
    "/content/data1/jpeg-192x192/test/0021f0d33.jpeg",
    "/content//jpeg-224x224/test/004b88e09.jpeg",
    "/content/drive/MyDrive/Hình/nho.jpg",
    "/content/drive/MyDrive/Hình/hd.jpg",
    "/content/drive/MyDrive/Hình/lili.jpg"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
class_indices = train_data_gen.class_indices
inverse_dict = {v: k for k, v in class_indices.items()}

for ax, img_path in zip(axes, test_images):
    if os.path.exists(img_path):
        img = Image.open(img_path).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_batch = np.expand_dims(img_array, 0)
        result = model.predict(img_batch)
        predict_class = np.argmax(result)
        predicted_label = inverse_dict[predict_class]
        ax.imshow(img)
        ax.set_title(f"{predicted_label}\nConf: {result[0][predict_class]:.2f}")
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, "Image not found", ha='center', va='center')
        ax.axis('off')

plt.tight_layout()
plt.savefig('test_predictions.png')
plt.close()

# Kiểm tra độ chính xác trên tập test
test_dir = os.path.join(data_path, "test1/")
if not os.path.exists(test_dir):
    print("Thư mục test/ không tồn tại! Vui lòng tạo thư mục và thêm dữ liệu.")
else:
    test_image_generator = ImageDataGenerator(rescale=1./255)
    test_data_gen = test_image_generator.flow_from_directory(
        directory=test_dir,
        batch_size=batch_size,
        shuffle=False,
        target_size=(im_height, im_width),
        class_mode='categorical'
    )

    test_steps = test_data_gen.n // batch_size
    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_data_gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, im_height, im_width, 3], [None, len(test_data_gen.class_indices)])
    ).repeat()

    print(f"Số mẫu test: {test_data_gen.n}")
    print(f"Test steps: {test_steps}")
    test_loss, test_accuracy = model.evaluate(test_dataset, steps=test_steps)
    print(f"Độ chính xác trên tập test: {test_accuracy * 100:.2f}%")
    print(f"Loss trên tập test: {test_loss:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot([test_loss], label='Test Loss', marker='o')
    plt.legend()
    plt.title('Test Loss')
    plt.xlabel('Evaluation')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot([test_accuracy], label='Test Accuracy', marker='o')
    plt.legend()
    plt.title('Test Accuracy')
    plt.xlabel('Evaluation')
    plt.ylabel('Accuracy')
    plt.savefig('test_metrics.png')
    plt.close()