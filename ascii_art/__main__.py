import argparse
from ascii_art import display_ascii_video


def _arg_parse():
    parser = argparse.ArgumentParser(
        description="Argument parser for video stream to ASCII. Press ESC key to terminate the video stream. CTRL+S will save a screenshot."
    )
    parser.add_argument(
        "--w", default=16, type=int, help="Width of each ASCII character, default 16."
    )
    parser.add_argument(
        "--h", default=12, type=int, help="Height of each ASCII character, default 12."
    )
    parser.add_argument(
        "--canny", action="store_true", help="Flag whether to use Canny Edge Detector."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _arg_parse()
    display_ascii_video(
        letter_width=args.w, letter_height=args.h, canny_edge=args.canny
    )
