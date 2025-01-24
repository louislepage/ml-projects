import argparse
import os

import cv2

from util import get_bboxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A simple parking spot detector and counter using cv2.")
    parser.add_argument("--video-path",
                        type=str,
                        default='data/parking_1920_1080_loop.mp4',
                        help="Path to video file."
                        )
    parser.add_argument("--mask-path",
                        type=str,
                        default='data/mask_1920_1080.png',
                        help="Path to mask file from which to derive parking spots."
                        )
    parser.add_argument("--data-dir",
                        type=str,
                        default='data/',
                        help="Path to data dir."
                        )
    return parser.parse_args()


def main():
    args = parse_args()

    # load video as capture
    cap = cv2.VideoCapture(args.video_path)

    # prep dirs
    os.makedirs(args.data_dir+"empty", exist_ok=True)
    os.makedirs(args.data_dir + "not_empty", exist_ok=True)

    # load mask
    mask = cv2.imread(args.mask_path, 0)

    # get connected_components from mask representing spots
    connected_components = cv2.connectedComponentsWithStats(mask, connectivity=4, ltype=cv2.CV_32S)

    # get bboxes
    bboxes = get_bboxes(connected_components)

    # read one frame
    ret, frame = cap.read()

    if not ret:
        print("There was an error reading the first frame to get cropped spots.")
        exit(1)

    for idx, spot in enumerate(bboxes):
        x1, y1, w, h = spot
        spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

        cv2.imshow('preview: pres y if is empty, n if not', spot_crop)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            out_path = f"{args.data_dir}not_empty/{idx}.jpg"
        elif key == ord('n'):
            out_path = f"{args.data_dir}empty/{idx}.jpg"
        else:
            print("Key not defined.")
            exit(1)

        cv2.imwrite(out_path, spot_crop)

    cap.release()



if __name__ == "__main__":
    main()
