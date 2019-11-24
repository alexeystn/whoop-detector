import cv2
import numpy as np
import time
import pyaudio
import wave
import os


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
            return (False, 0)
        if (lap_time > self.max_time) or (not self.started):
            lap_time = None
            self.started = True
        self.previous_timestamp = current_timestamp
        return (True, lap_time)
    

class Beeper(pyaudio.PyAudio):
  
    wave_file = None
    stream = None

    def callback(self, in_data, frame_count, time_info, flag):
        audio_data = self.wave_file.readframes(frame_count)
        return (audio_data, pyaudio.paContinue)
    
    def __init__(self, filename):
        pyaudio.PyAudio.__init__(self)
        w = wave.open(filename)
        self.stream = self.open(format=self.get_format_from_width(w.getsampwidth()),
                             channels=w.getnchannels(),
                             rate=w.getframerate(),
                             output=True,
                             stream_callback=self.callback)
        self.stream.stop_stream()
        self.wave_file = w

    def beep(self):
        self.stream.stop_stream()
        self.wave_file.rewind()
        self.stream.start_stream()  

    def close(self):
        self.wave_file.close()
        self.stream.close()
        self.terminate
        

class Capturer(cv2.VideoCapture):

    write_enabled = False

    def __init__(self, camera_id=0, write_to_file=False):
        cv2.VideoCapture.__init__(self, camera_id)
        self.set(cv2.CAP_PROP_FPS, 25)
        ret, img = self.read()
        ret, img = self.read()
        if not ret:
            print('Cannot capture video from camera')
        self.resolution = img.shape[:2]
        if write_to_file:
            fourcc = cv2.VideoWriter_fourcc('H','2','6','4')
            res = (img.shape[1], img.shape[0])
            num = 0
            while num < 10000:
                filename = 'video/{0:04d}.mp4'.format(num)
                if not os.path.exists(filename):
                    print('Writing to file <{0}>\n'.format(filename))
                    break
                num += 1
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


class Detector:

    resolution = (360, 270)
    buffer_length = 5
    
    def __init__(self):
        self.pointer = 0
        self.buffer = np.zeros((self.buffer_length, self.resolution[1],
                                self.resolution[0]), dtype='uint8')
        self.history = np.zeros((self.resolution[0], 1))

    def put_image(self, img):
        shape = self.buffer.shape
        self.pointer += 1
        if self.pointer == self.buffer_length:
            self.pointer = 0
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, self.resolution)
        img_gray = cv2.blur(img_gray, (10, 10))
        if not self.buffer.any(): 
            self.buffer[:] = img_gray
        self.buffer[self.pointer,:,:] = img_gray
        self.img_color_last = cv2.resize(img, self.resolution)

    def estimate_movement(self):
        plot_scale = 30
        threshold_difference_level = 7
        threshold_points_number = 1000
        
        img_smoothed = np.mean(self.buffer, axis=0).astype('uint8')
        img_diff = self.buffer[self.pointer,:,:].astype('float') - img_smoothed.astype('float')
        img_diff = np.abs(img_diff)
        mask = img_diff >= threshold_difference_level
        n_detected_points = np.count_nonzero(mask)
        result = n_detected_points > threshold_points_number
        
        img_output = self.img_color_last.copy()
        img_output[mask] = (0, 255, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_output, str(n_detected_points), (10,20), font, 0.5, [255,255,255], 2, cv2.LINE_AA)
        cv2.putText(img_output, str(n_detected_points), (10,20), font, 0.5, [0,0,0], 1, cv2.LINE_AA)

        point_y = np.log((n_detected_points / threshold_points_number)  * (np.e - 1) + 1)
        self.history[:-1] = self.history[1:]
        self.history[-1] = point_y
  
        y = self.resolution[1] - np.floor(self.history * plot_scale) - 1
        for i in range(len(self.history)-1):
            cv2.line(img_output, (i,y[i]), (i+1,y[i+1]), (0,255,255), 1)
        y = img_output.shape[0] - plot_scale
        cv2.line(img_output, (0,y), (img_output.shape[1],y), (0,0,255), 1)

        return (result, img_output)



timer = Timer()
beeper = Beeper('beep.wav')
capturer = Capturer(camera_id=1, write_to_file=False)
detector = Detector()

beeper.beep()

while True:

    img = capturer.get_frame()
    detector.put_image(img)
    detection_result, img_output = detector.estimate_movement()

    if detection_result:
        timer_result, lap_time = timer.put_event()
        if timer_result:
            beeper.beep()
            if lap_time:
                print('{0:7.3f} \n'.format(lap_time))
            else:
                print(' Start \n')   

    cv2.imshow('Whoop Detector', img_output)
    key = cv2.waitKey(1)
    if key == 27:
        break

beeper.close()
capturer.close()

