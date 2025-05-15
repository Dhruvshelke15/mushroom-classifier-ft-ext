import pandas as pd
import argparse
import tensorflow as tf
import os
import cv2
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model, Model

def decode_img(img_path, img_height, img_width):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0
    return img.numpy()

def get_images_labels(df, class_list, img_height, img_width):
    images, labels = [], []
    label_map = {name: idx for idx, name in enumerate(class_list)}
    for _, row in df.iterrows():
        label = row['label']
        img_path = row['image_path']
        if not os.path.exists(img_path):
            continue
        img = decode_img(img_path, img_height, img_width)
        images.append(img)
        labels.append(label_map[label])
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning or SVM Test")
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'svm'], help='Model type: cnn or svm')
    parser.add_argument('--model', type=str, default='model1_softmax_boosted.h5', help='Saved CNN model')
    parser.add_argument('--svm_model', type=str, default='svm_model.joblib', help='Saved SVM model')
    parser.add_argument('--test_csv', type=str, default='mushrooms_test.csv', help='CSV file with test images and labels')
    parser.add_argument('--img_height', type=int, default=160)
    parser.add_argument('--img_width', type=int, default=160)

    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv)
    class_list = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']

    test_images, test_labels = get_images_labels(test_df, class_list, args.img_height, args.img_width)

    if args.model_type == 'cnn':
        model = load_model(args.model)
        loss, acc = model.evaluate(test_images, tf.keras.utils.to_categorical(test_labels), verbose=2)
        print('Test model accuracy: {:5.2f}%'.format(100 * acc))
    else:
        # load CNN model only for feature extraction
        cnn_model = load_model(args.model)
        feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer(index=-3).output)
        features = feature_extractor.predict(test_images)

        # load trained SVM
        svm_model = joblib.load(args.svm_model)
        preds = svm_model.predict(features)
        acc = accuracy_score(test_labels, preds)
        print("SVM model accuracy: {:5.2f}%".format(100 * acc))
