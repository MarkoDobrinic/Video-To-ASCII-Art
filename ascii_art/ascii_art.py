import sys
import cv2
import numpy as np
from numba import jit

ASCII_CHARS = (
    """$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`'."""
)


def setup_video_capture(height=480, width=640, fps=30) -> cv2.VideoCapture:
    """Create VideoCapture instance and set width and height of the display.

    Args:
        height (int, optional): Height of the video display. Defaults to 720.
        width (int, optional): Width of the video display. Defaults to 960.
        fps (int, optional): Limit FPS of video stream. Defaults to 30.

    Returns:
        cv2.VideoCapture: Instance of cv2.VideoCapture
    """
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_FPS, fps)
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return vc


def display_ascii_video(
    letter_width: int, letter_height: int, window_name="camera_capture", canny_edge=True
) -> None:
    """Displays webcam video feed as ASCII art. `letter_width` and `letter_height`
    should be set to a ratio of 4:3 (16, 12) or (4, 3).

    Args:
        letter_width (int): Width of each ASCII character
        letter_height (int): Height of each ASCII character
        window_name (str, optional): Name of CV2 window. Defaults to "camera_capture".
        canny_edge (bool, optional): Use Canny Edge detector to simplify contours. Defaults to True.
    """
    try:
        ascii_images = map_ascii_characters(
            letter_width, letter_height, characters=ASCII_CHARS
        )
        cv2.namedWindow(window_name)
        vc = setup_video_capture()

        if vc.isOpened():
            rval, frame = vc.read()
        else:
            rval = False

        while rval:
            if canny_edge:
                transformed_frames = canny_edge_detection(frame)
            else:
                transformed_frames = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ascii_frame = convert_to_ascii(
                transformed_frames, ascii_images, letter_height, letter_width
            )
            cv2.imshow(window_name, ascii_frame)
            rval, frame = vc.read()
            key = cv2.waitKey(20)
            if key == 27:
                break

        cv2.destroyAllWindows()
        vc.release()

    except cv2.error as e:
        print("Failed to capture video stream: " + e)
        sys.exit()


def canny_edge_detection(frame: np.ndarray) -> np.ndarray:
    """Return frames processed by Canny Edge detector.

    Args:
        frame (np.ndarray): Input video frame.

    Returns:
        np.ndarray: Video frame processed by the Canny Edge detector.
    """
    frame = cv2.flip(frame, 1)
    gb = cv2.GaussianBlur(frame, (5, 5), 0)
    can = cv2.Canny(gb, 120, 30)
    return can


@jit(nopython=True)
def convert_to_ascii(
    frame: np.ndarray, ascii_images: np.ndarray, box_height=3, box_width=4
) -> np.ndarray:
    """Convert frames to ASCII characters.

    Args:
        frame (np.ndarray): Input video frames.
        ascii_images (np.ndarray): Array of ASCII characters converted to images.
        box_height (int, optional): Height of window of character region. Defaults to 3.
        box_width (int, optional): Width of window of character region. Defaults to 4.

    Returns:
        np.ndarray: Converted video frame to video frame in ASCII art.
    """
    h, w = frame.shape
    for i in range(0, h, box_height):
        for j in range(0, w, box_width):
            window = frame[i : i + box_height, j : j + box_width]
            avg_val = np.uint8(
                np.floor(np.sum(window) / (window.shape[0] + window.shape[1]))
            )
            ascii_image_idx = np.uint8(
                np.floor(np.interp(avg_val, [0, 255], [0, len(ascii_images) - 1]))
            )
            window[:, :] = ascii_images[::-1][ascii_image_idx]
    return frame


def map_ascii_characters(
    letter_width: int, letter_height: int, characters=ASCII_CHARS
) -> np.ndarray:
    """Maps each ASCII character to an image.

    Args:
        letter_width (int): Width of each character array.
        letter_height (int): Height of each character array.
        characters (str, optional): Characters to use for ASCII art representation. Defaults to ASCII_CHARS.

    Returns:
        np.ndarray: Array of ASCII characters converted to images.
    """
    imgs = []
    for char in characters:
        img = np.zeros((letter_height, letter_width), np.uint8)
        img = cv2.putText(img, char, (0, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)
        imgs.append(img)
    return np.stack(imgs)
