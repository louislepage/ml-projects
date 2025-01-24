import argparse
import os
import pickle

import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A simple parking spot detector and counter using cv2.")
    parser.add_argument("--model-path",
                        type=str,
                        default='data/custom_model.p',
                        help="Path to model which performs empty classification on each parking spots."
                        )
    parser.add_argument("--data-dir",
                        type=str,
                        default='data/',
                        help="Path to data dir."
                        )
    return parser.parse_args()


def main():
    args = parse_args()
    # load data and labels and flip for more data
    print(f"Loading data from {args.data_dir}...")
    data, labels = load_data_and_labels(args.data_dir)

    # split data
    x_tr, x_te, y_tr, y_te = train_test_split(data, labels, test_size=.2, shuffle=True, stratify=labels)

    # classifier
    print(f"Training SVM on training set with grid search...")
    svc = SVC()
    parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
    grid_search = GridSearchCV(svc, parameters)
    grid_search.fit(x_tr, y_tr)

    model = grid_search.best_estimator_

    print("Testing best classifier...")
    y_pred = model.predict(x_te)
    acc_score = accuracy_score(y_te, y_pred)
    print(f"Accuracy: {acc_score}")

    print(f"Saving model to {args.model_path}...")
    pickle.dump(model, open(args.model_path, 'wb'))
    print("Done")







def load_data_and_labels(data_dir):
    categories = ['empty', 'not_empty']

    data = []
    labels = []

    for idx, cat in enumerate(categories):
        for file in os.listdir(os.path.join(data_dir, cat)):
            if not file.endswith('.jpg'):
                continue
            img_path = os.path.join(data_dir, cat, file)
            img = imread(img_path)
            img = resize(img, (15, 15))
            # normal image
            data.append(img.flatten())
            labels.append(idx)
            # flip horizontally
            data.append(img[:, ::-1].flatten())
            labels.append(idx)
            # flip vertically
            data.append(img[::-1, :].flatten())
            labels.append(idx)
            # flip vertically and horizontally
            data.append(img[::-1, ::-1].flatten())
            labels.append(idx)

    data = np.asarray(data)
    labels = np.asarray(labels)

    return data, labels


if __name__ == "__main__":
    main()