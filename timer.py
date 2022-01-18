import cv2
import numpy as np
import time
import playsound
import os
import sys


class Timer:

    min_time = 5
    max_time = 60
    started = False
    
    def __init__(self):
        self.started = False
        self.previous_timestamp = time.time()

    def put_event(self):
        current_timestamp = time.time()
        lap_time = current_timestamp - self.previous_timestamp
        if lap_time < self.min_time:
            return False, 0
        if (lap_time > self.max_time) or (not self.started):
            lap_time = None
            self.started = True
        self.previous_timestamp = current_timestamp
        return True, lap_time


class Capturer(cv2.VideoCapture):

    write_enabled = False

    def __init__(self, camera_id=0, write_to_file=False):
        cv2.VideoCapture.__init__(self, camera_id)
        self.set(cv2.CAP_PROP_FPS, 25)
        self.read()
        time.sleep(0.5)
        ret, img = self.read()
        if not ret:
            print('Cannot capture video from camera')
            input()
            sys.exit(-1)
        self.resolution = img.shape[:2]
        if write_to_file:
            fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
            res = (img.shape[1], img.shape[0])
            for num in range(10000):
                filename = 'video/{0:04d}.mp4'.format(num)
                if not os.path.exists(filename):
                    print('Writing to file <{0}>\n'.format(filename))
                    break
            self.writer = cv2.VideoWriter(filename, fourcc, 25, res)
            self.write_enabled = True

    def get_frame(self):
        ret, img = self.read()
        if self.write_enabled:
            self.writer.write(img)
        return img
    
    def close(self):
        if self.write_enabled:
            self.writer.release()
        self.release()
        cv2.destroyAllWindows()


def beep():
    playsound.playsound('beep.wav', block=False)
    return


class Detector:

    resolution = (360, 270)
    buffer_length = 5
    paused = False
    img_color_last = None
    
    def __init__(self):
        self.pointer = 0
        self.buffer = np.zeros((self.buffer_length, self.resolution[1],
                                self.resolution[0]), dtype='uint8')
        self.history = np.zeros((self.resolution[0], 1))

    def put_image(self, img):
        self.pointer += 1
        if self.pointer == self.buffer_length:
            self.pointer = 0
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, self.resolution)
        img_gray = cv2.blur(img_gray, (20, 20))
        if not self.buffer.any(): 
            self.buffer[:] = img_gray
        self.buffer[self.pointer, :, :] = img_gray
        self.img_color_last = cv2.resize(img, self.resolution)

    def estimate_movement(self):
        plot_scale = 30
        threshold_difference_level = 7
        threshold_points_number = 1000

        img_smoothed = np.mean(self.buffer, axis=0).astype('uint8')
        img_diff = self.buffer[self.pointer, :, :].astype('float') - img_smoothed.astype('float')
        img_diff = np.abs(img_diff)
        mask = img_diff >= threshold_difference_level
        n_detected_points = np.count_nonzero(mask)
        result = n_detected_points > threshold_points_number
        
        img_output = self.img_color_last.copy()

        if result:
            x_edges, y_edges = np.where(mask)
            p0 = (np.min(y_edges)-5, np.min(x_edges)-5)
            p1 = (np.max(y_edges)+5, np.max(x_edges)+5)
            # if p1[0]-p0[0] < 200 and p1[1]-p0[1] < 200:
            cv2.rectangle(img_output, p0, p1, color=(0, 255, 0), thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_output, str(n_detected_points), (10, 20), font, 0.5, [255, 255, 255], 2, cv2.LINE_AA)
        cv2.putText(img_output, str(n_detected_points), (10, 20), font, 0.5, [0, 0, 0], 1, cv2.LINE_AA)

        point_y = np.log((n_detected_points / threshold_points_number) * (np.e - 1) + 1)
        self.history[:-1] = self.history[1:]
        self.history[-1] = point_y
  
        y = self.resolution[1] - np.floor(self.history * plot_scale) - 1
        for i in range(len(self.history)-1):
            cv2.line(img_output, (i, int(y[i])), (i+1, int(y[i+1])), (0, 255, 255), 1)
        y = img_output.shape[0] - plot_scale
        cv2.line(img_output, (0, y), (img_output.shape[1], y), (0, 0, 255), 1)

        if self.paused:
            result = False
            cv2.rectangle(img_output, (50, 50), (60, 60), color=(0, 300, 0), thickness=-1)

        return result, img_output


def main():
    timer = Timer()
    capturer = Capturer(camera_id=0, write_to_file=False)
    detector = Detector()

    cv2.namedWindow('Whoop Detector', cv2.WINDOW_NORMAL)
    beep()

    while True:

        img = capturer.get_frame()
        detector.put_image(img)
        detection_result, img_output = detector.estimate_movement()

        if detection_result:
            timer_result, lap_time = timer.put_event()
            if timer_result:
                beep()
                if lap_time:
                    print('{0:7.3f}'.format(lap_time))
                else:
                    print('Start')

        cv2.imshow('Whoop Detector', img_output)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            detector.paused = not detector.paused

    capturer.close()


if __name__ == '__main__':
    main()
