import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import threading
import os

class ImageBatch:
    def __init__(self, data_path, training_ratio=1, ext='npy', on_sort=False):
        self._DATA_LIST = glob.glob(data_path)
        self._DATA_LIST.sort(key=self._sort_path) if on_sort == True else None
        self._N_DATA = len(self._DATA_LIST)
        self._TRAINING_RATIO = training_ratio
        self._TRAIN_DATA_LIST = self._DATA_LIST[:int(self._N_DATA * self._TRAINING_RATIO)]
        self._TEST_DATA_LIST = self._DATA_LIST[int(self._N_DATA * self._TRAINING_RATIO):]
        self.N_TRAIN_DATA = len(self._TRAIN_DATA_LIST)
        self.N_TEST_DATA = self._N_DATA - self.N_TRAIN_DATA
        self._step = 0  # step is number of calling images_next_batch
        self._stride = 0  # batch_size/num_thread (for threading)
        self._total_step = 0  # total_batch == 1 epoch
        self._batch = 0
        self._load_func = {'png': cv2.imread, 'npy': np.load}
        self._load = self._load_func[ext]

    def _sort_path(self,path):
        return int(os.path.basename(path)[:-4])

    def _thread_worker(self, batch_data_list, start, on_norm=False):
        for i in range(len(batch_data_list)):
            data = batch_data_list[i]
            image = self._load(data)
            image = image.astype(dtype=np.float32)
            if on_norm != False:
                image = min_max_norm(image)
            self._batch[start+i] = image

    def next_batch(self, batch_size, num_thread=1, training=True, on_norm=False, astype=None):
        if batch_size < num_thread:
            num_thread = batch_size
        elif num_thread < 1:
            num_thread = 1

        if self._step == 0:
            self._batch_dict = {}
            self._stride = batch_size // num_thread
            self._total_step = self.N_TRAIN_DATA//batch_size if (self.N_TRAIN_DATA % batch_size) ==0 \
                                else (self.N_TRAIN_DATA//batch_size) + 1

        data_start = self._step * batch_size if self._step+1 != self._total_step else -batch_size
        data_end = data_start + batch_size if self._step+1 != self._total_step else None
        data_list = self._TRAIN_DATA_LIST[data_start:data_end] if training == True else self._TEST_DATA_LIST[:batch_size]

        self._batch = [0 for i in range(batch_size)]
        threads = []
        for i in range(num_thread):
            start = i*self._stride
            end = start+self._stride if (i + 1) != num_thread else None
            p = threading.Thread(target=self._thread_worker, args=(data_list[start:end],
                                                                   start,
                                                                   on_norm))
            threads.append(p)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if astype =="array":
            self._batch = np.array(self._batch)
        self._step = self._step+1 if self._step+1!=self._total_step else 0  # check step is the last or not

        return self._batch

    def train_shuffle(self, indicese):
        self._TRAIN_DATA_LIST = [self._TRAIN_DATA_LIST[index] for index in indicese]

    def test_shuffle(self, indicese):
        self._TEST_DATA_LIST = [self._TEST_DATA_LIST[index] for index in indicese]

class Time:
    def require_time(start, end, count):

        time = end - start
        total_time = time * count

        hour = total_time // 3600
        minute = int(total_time % 60)

        print("Total required time: {}hour {}minute".format(hour, minute))

def min_max_norm(images, mapping=1):
    '''
     :param norm_images: single or multiple image data.
                     shape is (height,width,n_channel) or (batch_size,height,width,n_channel)
     :return: return min_max_normalized signle or multiple data.
               shape of output is the same as shape of input
    '''
    norm_images = np.copy(images)
    shape = norm_images.shape
    norm_images = norm_images.astype(np.float32)
    norm_images = norm_images if len(shape) != 3 else norm_images.reshape((-1, *shape))
    batch_size = norm_images.shape[0]

    for i in range(batch_size):
        max = np.max(norm_images[i])
        min = np.min(norm_images[i])
        norm_images[i] = (norm_images[i] - min) / (max - min)

    norm_images = norm_images if len(shape) != 3 else norm_images.reshape(shape)
    return norm_images * mapping

def display(images_list, title=None, figsize=None, axis_off=False, size_equal=False,
            gridspec=(None, None), zoom_coordinate=(None, None, None, None)):
    n_images = len(images_list)
    batch_size = len(images_list[0])

    rows = n_images
    columns = batch_size
    fig, axis = plt.subplots(rows, columns, figsize=figsize, gridspec_kw={'wspace': gridspec[0], 'hspace': gridspec[1]})

    if rows == 1 and columns == 1:
        image = images_list[0][0]
        height, width, n_channel = image.shape
        size = [height, width, n_channel] if n_channel == 3 else [height, width]

        image = image.reshape(*size)
        image = image.astype(np.uint8)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        if rows == 1 or columns == 1:
            index = 0
            for i in range(rows):
                for j in range(columns):
                    image = images_list[i][j]
                    height, width, n_channel = image.shape
                    size = [height, width, n_channel] if n_channel == 3 else [height, width]
                    image = image.reshape(*size)

                    image = image.astype(np.uint8)
                    axis[index].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) if len(size) == 3 else axis[index].imshow(image)

                    if zoom_coordinate[0] != None:
                        axis[index].set_xlim(zoom_coordinate[0], zoom_coordinate[1])
                        axis[index].set_ylim(zoom_coordinate[3], zoom_coordinate[2])
                    if size_equal == True: axis[index].set_aspect('auto')
                    if axis_off == True: axis[index].set_axis_off()
                    '''
                    if title[i] != None:
                      axis[i].set_title(title[i])
                    '''
                    index += 1

        else:
            for i in range(rows):
                for j in range(columns):
                    image = images_list[i][j]
                    height, width, n_channel = image.shape
                    size = [height, width, n_channel] if n_channel == 3 else [height, width]
                    image = image.reshape(*size)

                    image = image.astype(np.uint8)
                    axis[i][j].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) if len(size) == 3 else axis[i][j].imshow(image)

                    if zoom_coordinate[0] != None and i % 2 == 1: axis[i][j].set_xlim(zoom_coordinate[0], zoom_coordinate[1])
                    if zoom_coordinate[0] != None and i % 2 == 1: axis[i][j].set_ylim(zoom_coordinate[3], zoom_coordinate[2])
                    if size_equal == True: axis[i][j].set_aspect('auto')
                    if axis_off == True: axis[i][j].set_axis_off()

                    '''
                    if title[i] !=None:
                        axis[i][0].set_title(title[i])
                  '''
    plt.show()

def psnr(x, y, peak=255):
    '''
    :param x: images
    :param y: another images
    :param peak: MAX_i peak. if int8 -> peak =255
    :return: return psnr value
    '''
    _max = peak
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    diff = (x-y).flatten('C')

    rmse = np.sqrt(np.mean(diff**2))
    result = 20 * np.log10(_max/rmse)
    return result