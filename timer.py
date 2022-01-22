import cv2
import numpy as np
import time
import datetime
import playsound
import os
import sys
import configparser


class Timer:

    def __init__(self, config):
        self.min_time = int(config['RACE']['min_lap_time'])
        self.max_time = int(config['RACE']['max_lap_time'])

        self.started = False
        self.previous_timestamp = time.time()

    def put_event(self):
        current_timestamp = time.time()
        lap_time = current_timestamp - self.previous_timestamp
        if lap_time < self.min_time:
            return False, 0

        if not self.started:
            lap_time = None
            self.started = True
        else:
            if lap_time > self.max_time:
                lap_time = None

        self.previous_timestamp = current_timestamp
        return True, lap_time

    def reset(self):
        self.started = False


class Logger:
    
    output_path = './logs/'
    file = None

    def __init__(self):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
            
        d = datetime.datetime.now() 
        filename = self.output_path + d.strftime('%Y%m%d_%H%M%S.txt')
        self.file = open(filename, 'w')

    def put(self, value):
        print(value)
        d = datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')
        self.file.write(d + '\t' + value + '\n')
        return 0

    def close(self):
        self.file.close()
    

class Capturer(cv2.VideoCapture):

    write_enabled = False

    def __init__(self, config, write_to_file=False):
        cv2.VideoCapture.__init__(self, int(config['DETECTION']['camera_id']))
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
                    self.writer = cv2.VideoWriter(filename, fourcc, 25, res)
                    self.write_enabled = True
                    break

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

    buffer_length = 5
    paused = False
    img_color_last = None
    
    def __init__(self, config):
        self.resolution = (int(config['SCREEN']['width']), int(config['SCREEN']['height']))
        self.sensitivity = int(config['DETECTION']['sensitivity'])
        self.pointer = 0
        self.buffer = np.zeros((self.buffer_length, self.resolution[1],
                                self.resolution[0]), dtype='uint8')
        self.history = np.zeros((self.resolution[0], 1))
        self.time_to_save = 0
        self.frame_counter = 0
        self.prev_detected_points_part = 0
        self.laps_list = []

    def toggle_pause(self):
        self.paused = not self.paused

    def decrease_sensitivity(self):
        if self.sensitivity > 1:
            self.sensitivity -= 1
            self.time_to_save = time.time() + 3

    def increase_sensitivity(self):
        if self.sensitivity < 10:
            self.sensitivity += 1
            self.time_to_save = time.time() + 3

    def check_config_status(self, config):
        if self.time_to_save:
            if self.time_to_save < time.time():
                self.time_to_save = 0
                config['DETECTION']['sensitivity'] = str(self.sensitivity)
                with open('config.ini', 'w') as configfile:
                    config.write(configfile)
                    print('Config saved')

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

    def put_lap(self, lap_time):
        self.laps_list.append(lap_time)

    def clear_laps(self):
        self.laps_list = []

    def estimate_movement(self):

        def apply_log_scale(values, sensitivity):
            return np.log(values * np.exp((sensitivity - 5)/3) * (np.e - 1) + 1)

        def print_text(image, text, pos, size):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, pos, font, size, [255, 255, 255], 3, cv2.LINE_AA)
            cv2.putText(image, text, pos, font, size, [0, 0, 0], 1, cv2.LINE_AA)

        threshold_difference_level = 7  # pixel brightness 0..255

        img_smoothed = np.mean(self.buffer, axis=0).astype('uint8')
        img_diff = self.buffer[self.pointer, :, :].astype('float') - img_smoothed.astype('float')
        img_diff = np.abs(img_diff)
        mask = img_diff >= threshold_difference_level
        n_detected_points = np.count_nonzero(mask)
        detected_points_part = n_detected_points / np.prod(self.resolution)
        # print('{0:.3f}'.format(detected_points_part))

        self.frame_counter += 1

        if self.frame_counter % 2 == 0:  # slow down the plot twice, keeping peak values
            new_history_value = np.max([self.prev_detected_points_part, detected_points_part])
            self.history[:-1] = self.history[1:]
            self.history[-1] = new_history_value
        self.prev_detected_points_part = detected_points_part

        result = apply_log_scale(detected_points_part, self.sensitivity) > 0.5
        
        img_output_video = self.img_color_last.copy()

        if result:
            x_edges, y_edges = np.where(mask)
            p0 = (np.min(y_edges)-5, np.min(x_edges)-5)
            p1 = (np.max(y_edges)+5, np.max(x_edges)+5)
            cv2.rectangle(img_output_video, p0, p1, color=(0, 255, 0), thickness=2)

        img_output_plot = np.zeros((self.resolution[1]//3, self.resolution[0], 3), dtype='uint8')

        y = img_output_plot.shape[0] // 2
        cv2.line(img_output_plot, (0, y), (img_output_plot.shape[1], y), (127, 127, 127), 1)

        y = (1 - apply_log_scale(self.history, self.sensitivity)) * self.resolution[1]//3
        y[y < 3] = 3
        y[y > self.resolution[1]//3-4] = self.resolution[1]//3-4
        for i in range(len(self.history)-1):
            cv2.line(img_output_plot, (i, int(y[i])), (i+1, int(y[i+1])), (255, 255, 255), 1, cv2.LINE_AA)

        if self.paused:
            result = False
            print_text(img_output_plot, '[Pause]', (self.resolution[0]//2 - 40, 40), 1)
        for i, lap in enumerate(self.laps_list):
            y = i * 20 + 30
            print_text(img_output_video, '{0:.2f}'.format(lap), (20, y), 0.5)
            if i > self.resolution[1]//20:
                break

        img_output = np.vstack((img_output_video, img_output_plot))

        return result, img_output


def main():

    config = configparser.ConfigParser()
    config.read('config.ini')
    timer = Timer(config)
    capturer = Capturer(config)
    detector = Detector(config)
    logger = Logger()

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
                    logger.put('{0:.2f}'.format(lap_time))
                    detector.put_lap(lap_time)
                else:
                    logger.put('Start')

        cv2.imshow('Whoop Detector', img_output)
        key = cv2.waitKey(1)
        if key == 27:  # esc
            break
        elif key == 32:  # space
            detector.toggle_pause()
            timer.reset()
        elif key == 13:  # enter
            detector.clear_laps()
        elif key == 45:  # minus
            detector.decrease_sensitivity()
        elif key == 61:  # plus
            detector.increase_sensitivity()
        detector.check_config_status(config)

    capturer.close()
    logger.close()


if __name__ == '__main__':
    main()
