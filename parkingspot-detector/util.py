import pickle
import cv2
import numpy as np
from skimage.transform import resize





def get_bboxes(connected_components):
    (tLabels, label_ids, values, centroid) = connected_components

    spots = []
    coef = 1

    # iter through components and retrieve info on bbox
    for i in range(tLabels):

        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        spots.append([x1, y1, w, h])

    return spots[1:]


def is_empty_spot_predict(spot_crop, model):

    # resize crop so every one is the same size
    spot_resized = resize(spot_crop, (15, 15, 3))
    # flatten and reshape into column vector
    flat_data = spot_resized.flatten().reshape(1, -1)

    y_pred = model.predict(flat_data)

    return y_pred == 0

def get_indices_with_significant_z_score(frame, prev_frame, bboxes, threshold=3) :

    # return no indices, if there is no previous frame
    if prev_frame is None:
        return []

    diffs = np.zeros(len(bboxes))
    for idx, spot in enumerate(bboxes):

        x1, y1, w, h = spot
        spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
        spot_crop_prev = prev_frame[y1:y1 + h, x1:x1 + w, :]
        diffs[idx] = np.abs(np.mean(spot_crop)-np.mean(spot_crop_prev))

    z_scores = (diffs - np.mean(diffs)) / np.std(diffs)

    return np.where(np.abs(z_scores) > threshold)[0]
