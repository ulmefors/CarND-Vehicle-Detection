import os
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import lib.window_slider as window_slider


def run_pipeline():
    window_slider.slide()


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
