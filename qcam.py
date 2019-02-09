import cv2
import threading
from queue import Queue
from time import time


class ClearingQueue(Queue):
    def __init__(self, maxsize=10):
        super().__init__(maxsize=maxsize)
        self.count = 0

    def put(self, item):

        with self.mutex:
            if self.maxsize > 0:
                if self.maxsize > self._qsize():
                    self._put(item)

    def put_clear(self, item):

        with self.mutex:
            if self.maxsize == self._qsize():
                self.queue.clear()
            self._put(item)


class Qcam:

    def __init__(self, src=0, frame_skip=False, frame_skip_num=1, fs_adaptive = False):

        """

        :param src: Source of the cam
        :param frame_skip: Boolean flag. True if you want to skip every nth frame (indicated by frame_skip_num
        :param frame_skip_num:
        :param fs_adaptive: Boolean flag. Set to True if you want frame skipping to be adaptive
        """
        self.frame_skip_num = frame_skip_num
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FPS, 1)
        self.stopped = False
        self.q = ClearingQueue(maxsize=3)
        self.grabbed, self.frame = self.stream.read()
        self.time_start = time()

    def start(self):

        threading.Thread(target=self.update, args=()).start()

    def update(self):
        fc = 0
        while True:
            if self.stopped:
                return

            if self.grabbed:
                if self.frame_skip_num:
                    fc += 1
                    if fc % self.frame_skip_num == 0:
                        # print('frameskip', fc)
                        fc = 0
                        continue
                self.q.put_clear(self.frame)
                self.grabbed, self.frame = self.stream.read()

            else:
                self.release_cam()
                continue
            self.grabbed, self.frame = self.stream.read()

    def read_from_queue(self):
        # print(1 / (time() - self.time_start))
        self.time_start = time()
        try:
            print('read')
            return True, self.q.get(block=True, timeout=1)

        except ValueError:
            print('value Exception')
            self.release_cam()
            return False, None

    def release_cam(self):
        self.stopped = True
        self.stream.release()
        print('cam released')

    def isOpen(self):
        return self.stream.isOpened()
