import cv2

class VideoCamera(object):
    def __init__(self):
        # 0 is usually the default built-in webcam
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # Release the camera when the object is destroyed
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None
        
        # Encode the frame into JPEG format
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()