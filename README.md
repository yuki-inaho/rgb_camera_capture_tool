To run this software, it is necessary to ready 
camera setting file(cfg/camera_parameter.toml)

# Installation
1.  Downloading this repository.
```
git clone https://github.com/yuki-inaho/see3cam_capture_tool.git
```

2. Instaling dependent python libraries.
```
cd see3cam_capture_tool
pip install -r requirements.txt
```

# Run capturing app
```
python capture.py
```

# Run capturing app with time-lapse
To capture a image per 10 minute, run below command
```
python capture.py --timelapse-mode --interval-minute 1
```
or
```
python capture.py -lapse -i 10
```