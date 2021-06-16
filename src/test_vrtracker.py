from time import sleep
from utils.ovr import VRTracker


tracker = VRTracker()
while True:
    if not tracker.Read():
        continue

    tracking_text  = '[HMD] X:{0.px:.3f}, Y:{0.py:.3f}, Z:{0.pz:.3f}\t'.format(tracker.hmd)
    tracking_text += '[L_CON] X:{0.px:.3f}, Y:{0.py:.3f}, Z:{0.pz:.3f}\t'.format(tracker.lcon)
    tracking_text += '[R_CON] X:{0.px:.3f}, Y:{0.py:.3f}, Z:{0.pz:.3f}'.format(tracker.rcon)
    print(tracking_text, end='\r', flush=True)
    sleep(0.025)