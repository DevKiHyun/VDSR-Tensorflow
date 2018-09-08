import tensorflow as tf
import os
import time
import numpy as np
import sys

from VDSR.util import ImageBatch
from VDSR.util import psnr
from VDSR.util import Time

def training(VDSR, config):
    training_ratio = 1
    main_data_path = '.'
    '''
    DATA SET PATH
    '''
    train_x_images_path = '{}/data/train_291_input/*.npy'.format(main_data_path)
    train_y_images_path = '{}/data/train_291_label/*.npy'.format(main_data_path)
    test_labels_path = '{}/data/Set5/y_ch/*.npy'.format(main_data_path)
    test_inputs_path = '{}/data/Set5/y_ch_2x/*.npy'.format(main_data_path)
    '''
    TRAIN SET(291) and shuffle
    '''
    train_x_images_batch = ImageBatch(train_x_images_path, training_ratio=1, on_sort=True)
    train_y_images_batch = ImageBatch(train_y_images_path, training_ratio=1, on_sort=True)

    shuffle_indicese = list(range(train_x_images_batch.N_TRAIN_DATA))
    np.random.shuffle(shuffle_indicese)
    train_x_images_batch.train_shuffle(shuffle_indicese)
    train_y_images_batch.train_shuffle(shuffle_indicese)
    '''
    TEST SET(SET 5) & preprocessing
    '''
    test_labels_batch = ImageBatch(test_labels_path, training_ratio=training_ratio, on_sort=True, ext='npy')
    test_inputs_batch = ImageBatch(test_inputs_path, training_ratio=training_ratio, on_sort=True, ext='npy')
    test_labels = test_labels_batch.next_batch(batch_size=5)
    test_inputs = test_inputs_batch.next_batch(batch_size=5)

    avg_bicubic_psnr_y_ch = 0
    for i in range(len(test_labels)):
        _psnr = psnr(test_labels[i], test_inputs[i], peak=1)
        avg_bicubic_psnr_y_ch += _psnr/5

    '''
    HYPERPARAMETER
    '''
    training_epoch = config.training_epoch
    batch_size = config.batch_size
    n_data = train_x_images_batch.N_TRAIN_DATA
    total_batch = n_data // batch_size if n_data%batch_size ==0 else (n_data//batch_size) +1
    total_iteration = training_epoch * total_batch

    # total_global_step = epoch*total_batch = n_iteration
    learning_rate = tf.train.exponential_decay(learning_rate=config.learning_rate, global_step=VDSR.global_step,
                                               decay_steps=80 * total_batch, decay_rate=0.1, staircase=True)

    VDSR.neuralnet()
    VDSR.optimize(learning_rate=learning_rate, grad_clip=config.grad_clip, on_grad_clipping=config.on_grad_clipping)
    VDSR.summary(learning_rate=learning_rate)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=tf_config)
    writer = tf.summary.FileWriter('./model/vdsr_result', sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    print("Total the number of Data : " + str(train_x_images_batch.N_TRAIN_DATA))
    print("Total Step per 1 Epoch: {}".format(total_batch))
    print("The number of Iteration: {}".format(total_iteration))

    for epoch in range(training_epoch):
        avg_cost = 0
        avg_vdsr_psnr_y_ch = 0
        for i in range(total_batch):
            start = time.time()

            batch_x = train_x_images_batch.next_batch(batch_size, num_thread=8, astype='array')
            batch_y = train_y_images_batch.next_batch(batch_size, num_thread=8, astype='array')
            batch_x = np.expand_dims(batch_x, axis=-1)
            batch_y = np.expand_dims(batch_y, axis=-1)

            summaries, _cost, _, g_step, lr = sess.run(
                [VDSR.summaries, VDSR.cost, VDSR.optimizer, VDSR.global_step, learning_rate],
                feed_dict={VDSR.X: batch_x, VDSR.Y: batch_y})

            writer.add_summary(summaries, g_step)
            avg_cost += _cost / total_batch if np.isinf(_cost) != True else 0

            end = time.time()
            if i == 5:
                Time.require_time(start, end, count=total_iteration - g_step)

        '''
        Evaluate VDSR performance
       '''
        for index in range(len(test_labels)):
            '''
            Y ch test average PSNR
           '''
            label_y = (test_labels[index].copy()*255).astype(np.uint8)
            input_y = test_inputs[index].copy()

            shape = input_y.shape
            result_input_y = input_y.reshape((1, *shape, 1))
            result_input_y = sess.run(VDSR.conv_net, feed_dict={VDSR.X: result_input_y})
            result_input_y = result_input_y.reshape(shape).astype(np.float32)
            result_input_y = np.clip(result_input_y, 0, 1)*255
            result_input_y = result_input_y.astype(np.uint8)

            _psnr = psnr(label_y, result_input_y, peak=255)
            print(_psnr)
            avg_vdsr_psnr_y_ch += _psnr / 5

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), 'Learning Rate: ', lr,
              '\nY_Ch AVG PSNR:: Bicubic: {:.9f} || VDSR: {:.9f}'.format(avg_bicubic_psnr_y_ch, avg_vdsr_psnr_y_ch))

        '''
        Do shuffle train_set for 1 epoch
        '''
        np.random.shuffle(shuffle_indicese)
        train_x_images_batch.train_shuffle(shuffle_indicese)
        train_y_images_batch.train_shuffle(shuffle_indicese)

    print("학습 완료!")
    save_path = '{}/model/VDSR.ckpt'.format(os.path.abspath('.'))
    saver.save(sess, save_path)
    print("세이브 완료")