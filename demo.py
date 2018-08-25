import argparse
import sys
import tensorflow as tf
import numpy as np
import cv2

import VDSR.vdsr as vdsr
from VDSR.util import ImageBatch
from VDSR.util import display
from VDSR.util import psnr

# Source: https://stackoverflow.com/questions/29100722/equivalent-im2double-function-in-opencv-python
def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float32) / info.max # Divide all values by the largest possible value in the datatype

def bicubic_sr(input, scale):
    height, width, _ = input.shape
    height = height - (height%scale)
    width = width - (width%scale)
    input = input[0:height, 0:width, :]

    ycrcb = cv2.cvtColor(input, cv2.COLOR_RGB2YCR_CB)
    y_ch = ycrcb[:,:,0].copy()
    y_ch = im2double(y_ch.astype(np.uint8))
    low_rs =  np.clip(cv2.resize(
                        cv2.resize(y_ch.copy(), None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_CUBIC),
                         (width, height), interpolation=cv2.INTER_CUBIC), 0, 1)

    ycrcb[:,:,0] = (low_rs*255).astype(np.uint8)
    rgb = np.clip(cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB),0,255)

    return rgb

def VDSR_sr(sess, VDSR, input, scale):
    height, width, _ = input.shape
    height = height - (height % scale)
    width = width - (width % scale)
    input = input[0:height, 0:width, :]

    ycrcb = cv2.cvtColor(input, cv2.COLOR_RGB2YCR_CB)
    y_ch = ycrcb[:, :, 0].copy()
    y_ch = im2double(y_ch.astype(np.uint8))
    low_rs = np.clip(cv2.resize(
        cv2.resize(y_ch.copy(), None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_CUBIC),
        (width, height), interpolation=cv2.INTER_CUBIC), 0, 1)

    vdsr_y_ch = sess.run(VDSR.conv_net, feed_dict={VDSR.X:np.expand_dims([low_rs], axis=-1)})[0]
    ycrcb[:, :, 0] = (vdsr_y_ch[:,:,0] * 255)
    rgb = np.clip(cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB), 0, 255)

    return rgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_channel', type=int, default=1, help='-')
    parser.add_argument('--batch_size', type=int, default=20, help='-')
    parser.add_argument('--image_index', type=int, default=1, help='-')
    parser.add_argument('--scale', type=int, default=2, help='-')
    parser.add_argument('--coordinate', type=int, default=[50, 50], help='-')
    parser.add_argument('--interval', type=int, default=30, help='-')
    args, unknown = parser.parse_known_args()

    test_y_images_path = './data/Urban100/HR/*.png'
    result_save_path = './result'
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

    input = labels[index]
    bicubic_output = bicubic_sr(input.copy(), scale=scale)
    VDSR_output = VDSR_sr(sess, VDSR, input.copy(), scale=scale)

    cv2.imwrite('{}/{}.png'.format(result_save_path, 'original'), input)
    cv2.imwrite('{}/{}.png'.format(result_save_path, 'bicubic'), bicubic_output)
    cv2.imwrite('{}/{}.png'.format(result_save_path, 'VDSR'), VDSR_output)

    print("Bicubic PSNR: ", psnr(input, bicubic_output))
    print("VDSR PSNR: ", psnr(input, VDSR_output))

    #input_list = [input, input[x_start:x_start+interval,  y_start:y_start+interval, :]]
    #bicubic_list = [bicubic_output, bicubic_output[x_start:x_start+interval,  y_start:y_start+interval, :]]
    #VDSR_list = [VDSR_output, VDSR_output[x_start:x_start+interval,  y_start:y_start+interval, :]]

    original_list = np.array([input, bicubic_output, VDSR_output])

    #zoom_list = np.array(original_list[:])
    display_list = np.array([original_list])
    display(display_list)

    #display_list = np.array([original_list, zoom_list)
    #display(display_list,  figsize = (5,5), axis_off=True, size_equal=True, gridspec=(0,0), zoom_coordinate=(150, 190, 100,260))



