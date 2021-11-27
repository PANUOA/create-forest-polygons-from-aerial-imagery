import sys
import cv2

image_name = "./data-z15/0-map.png"


def lm_get_image_size(file_name):
    print('load %s' % file_name)
    img = cv2.imread(file_name)
    sp = img.shape
    print(sp)
    sz1 = sp[0]  # height(rows) of image
    sz2 = sp[1]  # width(colums) of image
    sz3 = sp[2]  # channels
    print('height: %d \nwidth: %d \nchannels: %d' % (sz1, sz2, sz3))
    return sp


def main():
    lm_get_image_size(image_name)


if __name__ == '__main__':
    sys.exit(main())
