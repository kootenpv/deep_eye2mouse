import time

import just
import cv2

from pynput.mouse import Controller

interfaces = {"mouse": None, "video_cap": None}


def get_interfaces():
    if interfaces["mouse"] is None:
        interfaces["mouse"] = Controller()
    if interfaces["video_cap"] is None:
        interfaces["video_cap"] = cv2.VideoCapture(0)
    return interfaces["mouse"], interfaces["video_cap"]


def set_mouse_position(x, y):
    interfaces["mouse"].position = (int(x), int(y))


def yield_images(interval=0.1):
    mouse, video_cap = get_interfaces()
    it = 0
    until_val = 0
    t1 = time.time()
    while True:
        # Capture frame-by-frame
        _, frame = video_cap.read()

        # flip horizontal
        frame = cv2.flip(frame, 1)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            until_val = it + 1000

        t2 = time.time()
        if t2 > t1 + interval and it < until_val:
            yield frame, it, list(mouse.position)
            it += 1
            t1 = t2

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the video_capture
    video_cap.release()
    cv2.destroyAllWindows()


def record(data_name, data_path="~/tracktrack/"):
    path = just.make_path(data_path + data_name + "/")
    offset = len(just.glob(path + "/im*.png"))
    for image, it, mouse_pos in yield_images():
        cv2.imwrite(path + "/im_{}.png".format(it + offset), image)
        just.append(mouse_pos, path + "/positions.jsonl")
