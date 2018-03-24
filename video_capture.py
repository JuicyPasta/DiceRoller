import argparse
import os

import numpy as np
import cv2

IMAGE_TYPES = ('.jpg', '.jpeg', '.png', '.bmp')
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)


def segment(frame, dice):
    # dice come out as solid white squares, everything else black
    if dice == 'blue':
        # blue dice in the roller, blue has a low b in LAB
        lab_b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))[2]
        ret, mask = cv2.threshold(lab_b, 100, 255, cv2.THRESH_BINARY_INV)
    else:
        # black dice on white paper
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # fill in the dots, a 6 was basically three strips and never looked like a square
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    outside = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(mask, outside, -1, 255, -1)

    # then knock off anything thin (text, shadows, cracks) that was stuck to a die
    k = int(21 * (frame.shape[0] / 960.0)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return mask


def scrub(frame):
    # demo1.avi got recorded with an old version of this that drew its boxes and die=N
    # right on the frames, so paint those back out before looking for dice
    frame = frame.copy()
    b, g, r = cv2.split(frame.astype(np.int16))
    green = ((g - np.maximum(r, b)) > 60).astype(np.uint8) * 255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(green)
    for x, y, w, h, area in stats[1:]:
        if w > 20 and h > 20 and y > 28:
            # the text sits on top of each box, cover it with the paper right above it
            left, right = max(x - 3, 0), min(x + 90, frame.shape[1])
            frame[y - 26:y + 2, left:right] = frame[y - 28, left:right]
            green[y - 26:y + 2, left:right] = 0
    return cv2.inpaint(frame, cv2.dilate(green, np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)


def count_pips(gray, contour, box, debug):
    x, y, w, h = box
    single_die = gray[y:y+h, x:x+w]

    # only look inside the die, and stay off the edges so the sides of the die dont count as dots
    die_mask = np.zeros(gray.shape, np.uint8)
    # hull, since a dot right on the edge leaves a notch in the outline
    cv2.drawContours(die_mask, [cv2.convexHull(contour)], -1, 255, -1)
    k = max(5, int(.12 * min(w, h))) | 1
    die_mask = cv2.erode(die_mask, np.ones((k, k), np.uint8))[y:y+h, x:x+w]
    if not die_mask.any():
        return 0

    # otsu on just the die so it splits dots from the die instead of die from the paper
    otsu_ret, th = cv2.threshold(single_die[die_mask > 0].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_ret, single_thresh = cv2.threshold(single_die, otsu_ret, 255, cv2.THRESH_BINARY_INV)
    single_thresh[die_mask == 0] = 255

    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 256
    params.filterByArea = True
    # dots scale with the die instead of hardcoding pixels for one camera
    params.minArea = 0.006 * w * h
    params.maxArea = 0.1 * w * h
    params.filterByCircularity = False
    params.filterByConvexity = False
    # dots are round, slivers of glare or edge are not
    params.filterByInertia = True
    params.minInertiaRatio = 0.35
    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(single_thresh)

    if debug:
        single_die_keypoints = cv2.drawKeypoints(single_thresh, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow("single_die", cv2.resize(single_die_keypoints, (200, 200)))

    return len(keypoints)


def detect_die(mask, gray, debug=False):
    img = cv2.medianBlur(mask, 7)

    canny = cv2.Canny(img, 100, 200)
    canny_thick = cv2.dilate(canny, np.ones((2, 2), np.uint8))

    if debug:
        cv2.imshow("canny_thick", canny_thick)

    # newer opencv only returns 2 things here instead of 3, [-2] works on both
    contours = cv2.findContours(canny_thick, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # die size limits were picked at 1280x960
    scale = gray.shape[0] * gray.shape[1] / (1280.0 * 960.0)

    dice = []
    for contour in contours:
        # signed area, so each edge only counts once (the inside of the line goes the other way)
        area = cv2.contourArea(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        box_ratio = w / float(h)

        if 2000 * scale < area < 20000 * scale and .7 < box_ratio < 1.4 and area > .45 * w * h:
            # skip it if we already have a die here
            cx, cy = x + w / 2.0, y + h / 2.0
            if any(dx <= cx <= dx + dw and dy <= cy <= dy + dh for dx, dy, dw, dh, n in dice):
                continue
            pips = count_pips(gray, contour, (x, y, w, h), debug)
            if 1 <= pips <= 6:
                dice.append((x, y, w, h, pips))

    return dice


def label(img, text, x, y, size):
    (tw, th), base = cv2.getTextSize(text, FONT, size, 2)
    cv2.rectangle(img, (x, y - th - base - 6), (x + tw + 8, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 4, y - base - 3), FONT, size, (255, 255, 255), 2, cv2.LINE_AA)


def annotate(frame, dice):
    for x, y, w, h, pips in dice:
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)
        label(frame, 'die=' + str(pips), x, y - 2, .8)
    label(frame, 'total=' + str(sum(d[4] for d in dice)), 20, 60, 1.4)
    return frame


def process(frame, dice_color, debug, clean=False):
    if clean:
        frame = scrub(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = segment(frame, dice_color)
    dice = detect_die(mask, gray, debug)
    print(' '.join(str(d[4]) for d in dice) or '-')
    return annotate(frame, dice)


def frames(inputs):
    # camera index, a video file, or a bunch of images
    if len(inputs) == 1 and inputs[0].isdigit():
        inputs = [int(inputs[0])]
    for source in inputs:
        if isinstance(source, str) and source.lower().endswith(IMAGE_TYPES):
            frame = cv2.imread(source)
            if frame is None:
                raise SystemExit("can't read " + source)
            yield source, frame
            continue
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit("can't open " + str(source))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield source, frame
        cap.release()


class Output(object):
    # -o something.avi writes a video, -o some/folder/ writes a png per frame
    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.writer = None
        self.count = 0
        if path and not path.lower().endswith(('.avi', '.mp4')):
            os.makedirs(path, exist_ok=True)

    def write(self, frame):
        if not self.path:
            return
        if self.path.lower().endswith(('.avi', '.mp4')):
            if self.writer is None:
                fourcc = cv2.VideoWriter_fourcc(*('mp4v' if self.path.lower().endswith('.mp4') else 'XVID'))
                self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (frame.shape[1], frame.shape[0]))
            self.writer.write(frame)
        else:
            cv2.imwrite(os.path.join(self.path, 'frame%04d.png' % self.count), frame)
        self.count += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()


def main():
    parser = argparse.ArgumentParser(description='finds dice and counts the dots')
    parser.add_argument('input', nargs='*', default=['0'], help='camera number, video file, or images (default: camera 0)')
    parser.add_argument('--dice', choices=['black', 'blue'], default='black', help='black dice on white paper, or the blue dice in the roller')
    parser.add_argument('-o', '--out', help='write the annotated output to a .avi/.mp4, or a folder of pngs')
    parser.add_argument('--headless', action='store_true', help="no windows, just write the output")
    parser.add_argument('--fps', type=float, default=20.0, help='fps for the output video')
    parser.add_argument('--scrub', action='store_true', help='paint out the old boxes burned into demo1.avi first')
    parser.add_argument('--debug', action='store_true', help='show the edge and single die windows too')
    args = parser.parse_args()

    out = Output(args.out, args.fps)
    still = all(i.lower().endswith(IMAGE_TYPES) for i in args.input)

    for source, frame in frames(args.input):
        frame = process(frame, args.dice, args.debug and not args.headless, args.scrub)
        out.write(frame)

        if not args.headless:
            cv2.imshow('dice', frame)
            # images wait for a key, video just keeps going
            if cv2.waitKey(0 if still else 1) & 0xFF == ord('q'):
                break

    out.close()
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
