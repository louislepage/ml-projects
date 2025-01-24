import argparse
import pickle

import cv2
import numpy as np

from util import get_bboxes, is_empty_spot_predict, get_indices_with_significant_z_score

# colors in cv2 BGR space
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)


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
    parser.add_argument("--model-path",
                        type=str,
                        default='data/custom_model.p',
                        help="Path to model which performs empty classification on each parking spots."
                        )
    parser.add_argument("--sample-rate",
                        type=int,
                        default=30,
                        help="Sample rate for detection, defaults to every 30 frames."
                        )
    parser.add_argument("--z-score-threshold",
                        type=float,
                        default=2.,
                        help="Threshold in z-score on simple image diffs above which to trigger actual prediction."
                        )
    return parser.parse_args()


def main():
    args = parse_args()

    # load video as capture
    cap = cv2.VideoCapture(args.video_path)

    # load mask
    mask = cv2.imread(args.mask_path, 0)

    # load model
    model = pickle.load(open(args.model_path, 'rb'))

    # get connected_components from mask representing spots
    connected_components = cv2.connectedComponentsWithStats(mask, connectivity=4, ltype=cv2.CV_32S)
    # get bboxes
    bboxes = get_bboxes(connected_components)

    spot_status = [None for _ in bboxes]
    frame_n = 0
    prev_frame = None
    ret = True
    # while frame was loaded, keep going
    while ret:
        # read frame
        ret, frame = cap.read()

        # only recheck every sample_rate frames
        if frame_n % args.sample_rate == 0:

            if spot_status[0] is None:
                # on the first run if we have no info on the states, just check all once
                signif_z_score_inds = np.arange(len(bboxes))
            else:
                # find spots where a simple diff compared to the last frame is more than 3 std-devs away from the mean
                signif_z_score_inds = get_indices_with_significant_z_score(frame,
                                                                           prev_frame,
                                                                           bboxes,
                                                                           args.z_score_threshold
                                                                           )

            # copy frame for z score computation
            prev_frame = frame.copy()

            # check spots where the approx by z score showed a change
            for idx in signif_z_score_inds:
                x1, y1, w, h = bboxes[idx]
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                spot_status[idx] = is_empty_spot_predict(spot_crop, model)

        # always plot spots according to last status
        for idx, spot in enumerate(bboxes):
            x1, y1, w, h = spot

            if spot_status[idx]:
                color = COLOR_GREEN
            else:
                color = COLOR_RED

            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

        # plot some info
        cv2.rectangle(frame, (120, 20), (750, 80), (255, 255, 255), -1)
        cv2.putText(frame, f"Free Parking Spots: {int(sum(spot_status))}/{len(spot_status)}", (140, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))

        cv2.putText(frame, f"Press q to quit", (1500, 1050),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))

        # show frame
        cv2.imshow('frame', frame)

        # quit if we hit q, check if key was pressed, get last 8 bits where ascii code is and check if it was q
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_n += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
