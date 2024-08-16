import os
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2
import pandas as pd
from collections import Counter
import shutil
import random
from ultralytics import YOLO
import torch
from crop import plot_boxes, add_spaces_to_string
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/predicts/allow-cors/', methods=['POST'])
@cross_origin()
def predicts():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})
    
    uploaded_file = request.files['file']

    if uploaded_file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        dir = "./yolotestsave"
        if os.listdir(dir):
            for filename in os.listdir(dir):
                file_path = os.path.join(dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                        print("files deleted")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print("files deleted")
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            print(f'{dir} is empty')
    except FileNotFoundError:
        print(f'{dir} does not exist')

    # Save the uploaded image
    directory_path = './assets/'
    filename = 'my_image.jpg'
    uploaded_file.save(os.path.join(directory_path, filename))
    img = cv2.imread("./assets/my_image.jpg")

    if img is None:
        return jsonify({'error': 'Failed to read the image'})

    # here we now use the actual image
    img = cv2.imread("./assets/my_image.jpg")
    img = cv2.resize(img, (500, 300))
    img = cv2.GaussianBlur(img, (1, 1), 0)
    if img is None:
        return jsonify({'error': 'Failed to read the image'})

    directory = "./yolotestsave/"
    proximity_list, sorted_class_names = plot_boxes("./assets/my_image.jpg", directory)
    print("The proximity list: ", proximity_list)
    # all_files = sorted(os.listdir(directory))
    all_files = os.listdir(directory)
    # Sort the file names using the custom sorting function
    sorted_file_names = sorted(all_files, key=extract_numbers)
    sorted_names = []
    # Print the sorted file names
    for file_name in sorted_file_names:
        print(file_name)
        sorted_names.append(file_name)

    if sorted_names == []:
        return jsonify({'error': 'Failed to read the image'})
    
    predictions = []

    if len(sorted_names) <= 1:
        result_string = ' '.join(sorted_class_names)
        predictions = add_spaces_to_string(result_string)
        print("predictions after using yolo model")
        print(predictions)
    else:
        model = load_model('dataModel.h5')

        for filename in sorted_names:
            if filename.endswith(".jpg"):
                # Construct full file path
                filepath = os.path.join(directory, filename)
                img1 = cv2.imread(filepath)
                img1 = cv2.resize(img1, (28, 28), interpolation=cv2.INTER_AREA)
                img1 = img1[np.newaxis, :, :, :]
                # Predict the class
                a = np.argmax(model.predict(img1), axis=1)
                alphabet_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h' , 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
                predicted_alphabet = alphabet_dict[int(a)]
                predictions.append(predicted_alphabet)
                print(predicted_alphabet)
        for i, add_space in enumerate(proximity_list):
            if add_space:
                predictions.insert(i+1, ' ')
                print("predictions after using my CNN model")
                print(predictions)

    i = 0
    print(predictions)
    if len(predictions) == 0:
        return jsonify({'result': 'Failed to read the image'})
    else:
        return jsonify({'result': list(predictions)})
    
def extract_numbers(filename):
    parts = filename.split('_')
    line = int(parts[1])
    segment = int(parts[3].split('.')[0])
    return (line, segment)

if __name__ == '__main__':
    app.run(debug=True, host="192.168.43.63", port=8000)