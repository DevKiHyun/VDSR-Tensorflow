import argparse
import tensorflow as tf
import numpy as np
import cv2
import os

import VDSR.vdsr as vdsr
from VDSR.util import ImageBatch
from VDSR.util import display
from VDSR.util import psnr

def modcrop(image, scale):
    height, width, _ = image.shape
    height = height - (height % scale)
    width = width - (width % scale)
    image = image[0:height, 0:width, :]

    return image

# Source: https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python
def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float32) / info.max # Divide all values by the largest possible value in the data type

def bicubic_sr(low_input, scale):
    upscaled_rgb = np.clip(cv2.resize(low_input.copy(), None, fx=1.0 * scale, fy=1.0 * scale, interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)

    return upscaled_rgb

def VDSR_sr(sess, VDSR, low_input, scale):
    upscaled_rgb = np.clip(cv2.resize(low_input.copy(), None, fx=1.0 * scale, fy=1.0 * scale, interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)

    upscaled_rgb = im2double(upscaled_rgb.astype(np.uint8))

    VDSR_output = sess.run(VDSR.output, feed_dict={VDSR.X: np.expand_dims(upscaled_rgb, axis=0)})
    VDSR_output = np.squeeze(VDSR_output, axis=0)
    VDSR_output *= 255

    return VDSR_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_channel', type=int, default=3, help='-')
    parser.add_argument('--batch_size', type=int, default=100, help='-')
    parser.add_argument('--image_index', type=int, default=7, help='-')
    parser.add_argument('--scale', type=int, default=2, help='-')
    parser.add_argument('--coordinate', type=int, default=[50, 50], help='-')
    parser.add_argument('--interval', type=int, default=30, help='-')
    args, unknown = parser.parse_known_args()

    test_y_images_path = './data/Urban100/HR/*.png'
    result_save_path = './test_result'
    if not os.path.exists(result_save_path): os.makedirs(result_save_path)

    labels_images = ImageBatch(test_y_images_path, training_ratio=1, on_sort=True, ext='png')
    labels = labels_images.next_batch(batch_size=args.batch_size)

    VDSR = vdsr.VDSR(args)
    VDSR.neuralnet()

    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, './model/VDSR.ckpt')

    index = args.image_index
    scale = args.scale
    x_start = args.coordinate[0]
    y_start = args.coordinate[1]
    interval = args.interval

    label = modcrop(labels[index].copy(), scale=scale)

    low_rs = np.clip(cv2.resize(label.copy(), None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_CUBIC), 0, 255)

    bicubic_output = bicubic_sr(low_rs.copy(), scale=scale)
    VDSR_output = VDSR_sr(sess, VDSR, low_rs.copy(), scale=scale)

    cv2.imwrite('{}/{}_{}x.png'.format(result_save_path, 'original', scale), label)
    cv2.imwrite('{}/{}_{}x.png'.format(result_save_path, 'low', scale), low_rs)
    cv2.imwrite('{}/{}_{}x.png'.format(result_save_path, 'bicubic', scale), bicubic_output)
    cv2.imwrite('{}/{}_{}x.png'.format(result_save_path, 'VDSR', scale), VDSR_output)

    print("Bicubic {}x PSNR: ".format(scale), psnr(label, bicubic_output))
    print("VDSR {}x PSNR: ".format(scale), psnr(label, VDSR_output))

    #input_list = [input, input[x_start:x_start+interval,  y_start:y_start+interval, :]]
    #bicubic_list = [bicubic_output, bicubic_output[x_start:x_start+interval,  y_start:y_start+interval, :]]
    #VDSR_list = [VDSR_output, VDSR_output[x_start:x_start+interval,  y_start:y_start+interval, :]]

    original_list = np.array([label, bicubic_output, VDSR_output])

    #zoom_list = np.array(original_list[:])
    display_list = np.array([original_list])
    display(display_list)

    #display_list = np.array([original_list, zoom_list)
    #display(display_list,  figsize = (5,5), axis_off=True, size_equal=True, gridspec=(0,0), zoom_coordinate=(150, 190, 100,260))