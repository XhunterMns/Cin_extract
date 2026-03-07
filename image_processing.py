import os
import sys
import cv2


def read_image(file_path):

    if not os.path.isabs(file_path):
        base_dir = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(base_dir, file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"OpenCV could not read the image (corrupt or unsupported format): {file_path}")

    return image

def to_greyscale(image):
    if image is None:
        raise ValueError("to_greyscale received None as image")

    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grey_image


def blur(image, kernel_size=(5, 5)):
    if image is None:
        raise ValueError("blur received None as image")

    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image


def threshold(image, threshold_value=127):
    if image is None:
        raise ValueError("threshold received None as image")

    thresh_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 9)
    return thresh_image

def preprocess_image(file_path):
    image = read_image(file_path)
    grey_image = to_greyscale(image)
    blurred_image = blur(grey_image)
    thresh_image = threshold(blurred_image)
    return thresh_image


if __name__ == '__main__':
    # Simple CLI for quick testing. Usage: python3 image_processing.py [path/to/image]
   # input_path = sys.argv[1] if len(sys.argv) > 1 else 'static/files/test.jpg'
    input_path = 'static/files/cin.jpg'
    try:
        out = preprocess_image(input_path)
        # write output next to input with a suffix
        if not os.path.isabs(input_path):
            base_dir = os.path.abspath(os.path.dirname(__file__))
            input_path_abs = os.path.join(base_dir, input_path)
        else:
            input_path_abs = input_path

        out_name = os.path.splitext(os.path.basename(input_path_abs))[0] + '_preprocessed.png'
        out_dir = os.path.dirname(input_path_abs)
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, out)
        print(f'Preprocessed image written to: {out_path}')
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)