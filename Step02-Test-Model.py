import os
from keras.utils import img_to_array, load_img
import tensorflow as tf 
import numpy as np
import cv2

# Modelin yüklenmesi
model = tf.keras.models.load_model("e:/temp/retinalOCT.h5")

# Kategorilerin yüklenmesi
source_folder = "E:/Data-sets/Retinal OCT Images/val"
categories = os.listdir(source_folder)
categories.sort()

print(categories)

numOfClasses = len(categories)
print(numOfClasses)


def prepare_image(path_for_image):
    """
    Bir görüntüyü işleme için önceden hazırlar.

    Args:
        path_for_image (str): Görüntünün dosya yolu.

    Returns:
        numpy.ndarray: İşlenmiş görüntü.
    """

    image = load_img(path_for_image, target_size=(224, 224), color_mode='grayscale')
    img_result = img_to_array(image)
    img_result = np.expand_dims(img_result, axis=0)

    img_result = img_result / 255.

    return img_result


# Tahmin

test_image_path = "E:/Data-sets/Retinal OCT Images/val/Normal/1.jpg"
image = prepare_image(test_image_path)

prediction = model.predict(image)
predicted_class = np.argmax(prediction)

print(f"Tahmin edilen sınıf: {categories[predicted_class]}")
