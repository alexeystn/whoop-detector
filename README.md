# Whoop Detector
Webcam based timing system for indoor micro quads.

### How it works
* Point your camera on the wall or on the floor.
* Launch script `detector.py`.
* When your whoop appears in the camera field-of-view, a new lap is detected.

### How to install
1) Get the latest version of [Python](https://www.python.org/downloads/).
2) Install additional modules. Type in command line: <br>
`pip3 install opencv-python` <br>
`pip3 install playsound`

### Controls
* Space — Pause detection <br>
* Plus/Minus — Adjust sensitivity <br>
* C — Clear results <br>
* P — Show/hide plot <br>
* Esc — Exit <br>

### Tips
* If you have more than one web camera in your system (e.g. laptop front cam + external USB cam), you may need to change `camera_id` parameter from 0 to 1 in `config.ini`.

---
Feel free to contact me in Telegram: [@AlexeyStn](https://t.me/AlexeyStn)
