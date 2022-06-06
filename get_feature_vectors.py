import pandas as pd
import numpy as np
import cv2
import math
import os
from PIL import Image
import pywt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# sacar vector caracteristico
# normalizar
# guardar en carpeta

random_seed = np.random.seed(42)

def resize_and_save_img(src, destination_path):
    """
    resize_and_save_img aplica un resize a la imagen en src y la guarda en destination_path.
    :param src: Dirección a la imagen a la que se le desea hacer resize.
    :param destination_path: Dirección en la cual se debe guardar la imagen procesada.
    :return: None.
    """
    original_img = cv2.imread(src)
    old_image_height, old_image_width, channels = original_img.shape
    new_image_width = 60        
    new_image_height = 60
    color = (255,255,255)

    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)

    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # Centrar imagen
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = original_img

    Image.fromarray(result).save(destination_path)

def generate_new_data():
    """
    generate_new_data se corre solo una vez. Junta la data de Train y Test en un solo directorio unificado.
    :return: None.
    """
    train_dir = "Data/Train/"
    test_dir = "Data/Test/"
    destination_dir = "Data_preprocesada/"

    for class_dir in os.listdir(train_dir):
        for train_img in os.listdir(train_dir+class_dir):
            resize_and_save_img(f"{train_dir}{class_dir}/{train_img}", f"{destination_dir}{class_dir}/{train_img}")

    test_info = pd.read_csv("Data/Test.csv")
    for i, test_img in enumerate(sorted(os.listdir(test_dir))):
        resize_and_save_img(f"{test_dir}{test_img}", f"{destination_dir}{test_info.ClassId[i]}/{test_img}")

def get_vector_from_image(image, iterations):
    """
    get_vector_from_image obtiene el vector característico de la imagen image
    :param image: Imagen en formato vector.
    :param iterations: Entero que indica la cantidad de veces que se aplica el wavelet a la imagen.
    :return LL: Vector característico sin la compresión a 1D.
    :return LL.flatten(): Vector característico en 1D.
    """
    LL, (LH, HL, HH) = pywt.dwt2(image, 'haar')
    for _ in range(iterations - 1):
        LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
    return LL, LL.flatten()

