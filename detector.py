import os
import sys
import time
import datetime
import configparser

import cv2
import numpy as np
import playsound


class Timer:

    def __init__(self, config):
        self.min_lap_time = int(config['RACE']['min_lap_time'])
        self.max_lap_time = int(config['RACE']['max_lap_time'])
        self.started = False
        self.start_time = time.time()
        self.previous_timestamp = 0
        self.best_lap = 0
        self.lap_counter = 0

    def put_event(self):

        result = {'event': False, 'lap_time': 0, 'best': False}

        current_timestamp = time.time()
        if current_timestamp - self.start_time < 1:
            return result

        lap_time = current_timestamp - self.previous_timestamp

        if lap_time > self.min_lap_time:
            result['event'] = True
            if not self.started:
                self.started = True
            else:
                if lap_time > self.max_lap_time:
                    result['lap_time'] = 0
                else:
                    result['lap_time'] = lap_time
                    self.lap_counter += 1
                    if (self.best_lap == 0) or (lap_time < self.best_lap):
                        self.best_lap = lap_time
                        if self.lap_counter > 1:
                            result['best'] = True
            self.previous_timestamp = current_timestamp

        return result

    def reset(self):
        self.best_lap = 0
        self.lap_counter = 0
        self.started = False


class Logger:

    output_path = './logs/'

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

    output_path = './video/'
    demo_path = './demo/demo.mp4'

    def __init__(self, config):
        self.write_enabled = int(config['DEBUG']['save_video'])
        # cv2.VideoCapture.__init__(self, int(config['CAMERA']['camera_id']))
        cv2.VideoCapture.__init__(self, self.demo_path)
        # self.set(cv2.CAP_PROP_FPS, 30) ???
        self.read()
        time.sleep(0.5)
        ret, img = self.read()
        if not ret:
            print('Cannot capture video from camera')
            input()
            sys.exit(-1)
        self.resolution = img.shape[:2]
        if self.write_enabled:
            if not os.path.exists(self.output_path):
                os.mkdir(self.output_path)
            res = (img.shape[1], img.shape[0])
            for num in range(10000):
                filename = self.output_path + 'video_{0:04d}.mp4'.format(num)
                if not os.path.exists(filename):
                    print('Writing to file: {0}\n'.format(filename))
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.writer = cv2.VideoWriter(filename, fourcc, 30, res)
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


class Beeper:

    def __init__(self, config):
        self.enabled = int(config['SOUND']['sound_on'])

    def beep(self):
        if self.enabled:
            playsound.playsound('beep.wav', block=False)
            return


class Display:

    window_name = 'Whoop Detector'

    def __init__(self, config):
        self.height = int(config['SCREEN']['height'])
        self.width = int(config['SCREEN']['width'])
        self.history = np.zeros((self.width,))
        self.prev_history_value = 0
        self.history_counter = 0
        self.line_height = 34
        self.laps_list = []
        self.max_lap_count = int(config['SCREEN']['height'])//self.line_height
        self.show_plot = int(config['DEBUG']['show_plot'])
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def toggle_plot(self):
        self.show_plot = int(not self.show_plot)

    def put_history(self, value):
        self.history_counter += 1
        if self.history_counter % 2 == 0:  # slow down the plot twice, keeping peak values
            new_history_value = np.max([self.prev_history_value, value])
            self.history[:-1] = self.history[1:]
            self.history[-1] = new_history_value
        self.prev_history_value = value

    def draw_history_plot(self, sensitivity):
        image = np.zeros((self.height//3, self.width, 3), dtype='uint8')
        y = image.shape[0] // 2
        cv2.line(image, (0, y), (image.shape[1], y), (127, 127, 127), 1)
        y = (1 - apply_log_scale(self.history, sensitivity)) * self.height // 3
        y[y < 3] = 3
        y[y > self.height // 3 - 4] = self.height // 3 - 4
        for i in range(len(self.history) - 1):
            cv2.line(image, (i, int(y[i])), (i + 1, int(y[i + 1])), (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def put_lap(self, arg):
        self.laps_list.append(['{0:.2f}'.format(arg['lap_time']), arg['best']])
        if len(self.laps_list) > self.max_lap_count:
            self.laps_list.pop(0)
            self.laps_list[0] = ['^', False]

    def clear_laps(self):
        self.laps_list = []

    def show(self, args):

        def print_text(image, text, pos, size, highlight=False):
            font = cv2.FONT_HERSHEY_SIMPLEX
            (w, h), baseline = cv2.getTextSize(text, font, size, 2)
            pos = (pos[0] - w // 2, pos[1] + h // 2)
            if highlight:
                color = [0, 255, 255]
            else:
                color = [255, 255, 255]
            cv2.putText(image, text, pos, font, size, color, 2, cv2.LINE_AA)

        def draw_laps(image, laps, line_height):
            height = len(laps) * line_height
            image[10:height + 10, 10:110, :] //= 2
            for i, lap in enumerate(laps):
                y = i * self.line_height + line_height // 2 + 10
                print_text(image, lap[0], (60, y), 0.7, lap[1])

        def draw_pause(image):
            (x, y) = (self.width // 2, self.height // 2)
            image[y - 22:y + 22, x - 70:x + 70, :] //= 2
            print_text(image, 'Pause', (x, y), 1)

        def draw_mask(image, mask):
            show_points = False
            show_rectangle = True
            if not np.any(mask):
                return
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            if show_points:
                green_image = np.zeros(image.shape, image.dtype)
                green_image[:, :] = (0, 255, 0)
                green_mask = cv2.bitwise_and(green_image, green_image, mask=mask)
                cv2.addWeighted(green_mask, 1, image, 1, 0, image)
            if show_rectangle:
                x_points, y_points = np.where(mask)
                p0 = (np.min(y_points), np.min(x_points))
                p1 = (np.max(y_points), np.max(x_points))
                cv2.rectangle(image, p0, p1, color=(0, 255, 0), thickness=2)

        args['image'] = cv2.addWeighted(args['image'], 0.9, args['image'], 0, 0)
        if self.show_plot:
            draw_mask(args['image'], args['mask'])

        draw_laps(args['image'], self.laps_list, self.line_height)

        self.put_history(args['estimation'])

        if args['paused']:
            draw_pause(args['image'])

        img_output = args['image']

        cv2.imshow(self.window_name, img_output)


class Detector:

    def __init__(self, config):
        self.paused = False
        self.output_resolution = (int(config['SCREEN']['width']), int(config['SCREEN']['height']))
        self.detection_resolution = (320, 240)
        self.sensitivity = int(config['DETECTION']['sensitivity'])
        self.smoothness = int(config['DETECTION']['smoothness'])
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=10)

    def toggle_pause(self):
        self.paused = not self.paused

    def decrease_sensitivity(self):
        if self.sensitivity > 1:
            self.sensitivity -= 1
        print('Sensitivity: {0}'.format(self.sensitivity))

    def increase_sensitivity(self):
        if self.sensitivity < 20:
            self.sensitivity += 1
        print('Sensitivity: {0}'.format(self.sensitivity))

    def estimate_movement(self, image):
        image = cv2.resize(image, self.output_resolution)
        image_small = cv2.resize(image, self.detection_resolution)
        image_small = cv2.addWeighted(image_small, 1.9, image_small, 0, 0)
        image_small = cv2.blur(image_small, (self.smoothness, self.smoothness))
        foreground_mask = self.background_subtractor.apply(image_small)
        num_detected_points = np.count_nonzero(foreground_mask)
        movement_ratio = num_detected_points / np.prod(self.detection_resolution)
        result = apply_log_scale(movement_ratio, self.sensitivity) > 0.5

        if self.paused:
            result = False

        return {'image': image,
                'estimation': movement_ratio,
                'mask': foreground_mask,
                'sensitivity': self.sensitivity,
                'paused': self.paused,
                'result': result}


class Debug:

    def __init__(self, config):
        self.previous_cycle_ts = time.perf_counter()
        self.tic_ts = 0
        self.toc_ts = 0
        self.pointer = 0
        self.processing_time_buffer = np.zeros((10,))
        self.cycle_time_buffer = self.processing_time_buffer.copy()
        self.enabled = int(config['DEBUG']['print_load'])

    def tic(self):
        self.tic_ts = time.perf_counter()

    def toc(self):
        self.toc_ts = time.perf_counter()

    def cycle(self):
        if not self.enabled:
            return
        current_cycle_ts = time.perf_counter()
        processing_time = self.toc_ts - self.tic_ts
        cycle_time = current_cycle_ts - self.previous_cycle_ts
        self.previous_cycle_ts = current_cycle_ts
        self.cycle_time_buffer[self.pointer] = cycle_time
        self.processing_time_buffer[self.pointer] = processing_time
        self.pointer += 1
        if self.pointer == len(self.cycle_time_buffer):
            self.pointer = 0
            avg_processing_time = np.mean(self.processing_time_buffer)
            avg_cycle_time = np.mean(self.cycle_time_buffer)
            print('DEBUG:  {0:4.0f} us  {1:5.2f}% load  {2:.1f} FPS'.format(
                avg_processing_time * 1e6,
                avg_processing_time / avg_cycle_time * 100,
                1 / avg_cycle_time))


class Config(configparser.ConfigParser):

    filename = 'config.ini'

    def __init__(self):
        super().__init__()
        self.read(self.filename)

    def save(self, display, detector):
        # update only adjustable settings
        self['DETECTION']['sensitivity'] = str(detector.sensitivity)
        self['DEBUG']['show_plot'] = str(display.show_plot)
        with open(self.filename, 'w') as configfile:
            self.write(configfile)
            print('Config saved')


def apply_log_scale(values, scale):
    return np.log(values * np.exp((scale - 5) / 3) * (np.e - 1) + 1)


def main():

    config = Config()
    capturer = Capturer(config)
    detector = Detector(config)
    display = Display(config)
    beeper = Beeper(config)
    logger = Logger()

    beeper.beep()

    cnt = 0
    display.show_plot = 1

    lap_frames = [7*25, 9*25, 11*25, 13*25, 15*25]
    lap_times = [21.52, 22.19, 19.37, 20.64, 18.80]
    lap_best = [False, False, True, False, True]

    # laps = {f: {'lap_time': l, 'best': b} for f, l, b in zip(lap_frames, lap_times, lap_best)}
    laps = [{'lap_time': l, 'best': b} for l, b in zip(lap_times, lap_best)]

    while True:
        display.put_lap(laps[3])
        display.put_lap(laps[0])
        display.put_lap(laps[4])
        img = cv2.imread('./Demo/screenHD.png')
        img = cv2.resize(img, (320, 240))
        img_blur = cv2.blur(img, (10, 10))
        mask = (img_blur[:, :, 1] < 150).astype('uint8')
        img[:, :, 0] += 10
        img = cv2.addWeighted(img, 1, img, 0, 0)
        args = {'image': img,
                'estimation': 0,
                'mask': mask,
                'sensitivity': 1,
                'paused': 0,
                'result': 0}
        display.show(args)
        cv2.waitKey()
        sys.exit(0)

    while True:
        img = capturer.get_frame()
        if img is None:
            break
        args = detector.estimate_movement(img)
        display.show(args)
        cv2.waitKey(30)
        cnt += 1
        if cnt > 6*25:
            display.show_plot = 0
        if cnt in laps.keys():
            display.put_lap(laps[cnt])
        if cnt == 1:
            cv2.waitKey()
            time.sleep(2)
            cv2.waitKey(500)

    capturer.close()
    logger.close()
    config.save(display, detector)


if __name__ == '__main__':
    main()
