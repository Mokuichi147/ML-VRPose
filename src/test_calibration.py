from utils.camera import WebCam


camera = WebCam()
camera.StartCalibration(10, 1.85, 7, 7, save_dir='calibration')
while camera.IsOpened():
    if not camera.Read():
        print('Frame acquisition failed', end='\r', flush=True)
        continue

    camera.Show('WebCam')

    if camera.Wait():
        break