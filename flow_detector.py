import cv2 as cv
import numpy as np
import scipy.interpolate
from scipy import signal
import scipy.ndimage.filters
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import time
from datetime import datetime


class CA_CFAR():
    def __init__(self, win_param, p_fa, rd_size):
        win_width = win_param[0]
        win_height = win_param[1]
        guard_width = win_param[2]
        guard_height = win_param[3]

        self.mask = np.ones((2 * win_height + 1, 2 * win_width + 1), dtype=bool)
        self.mask[win_height - guard_height:win_height + 1 + guard_height,
        win_width - guard_width:win_width + 1 + guard_width] = 0

        N = ((2 * win_height + 1) * (2 * win_width + 1)) - ((2 * guard_width + 1) * (2 * guard_height + 1))
        self.scaling_factor = N * ((p_fa ** (-1 / N)) - 1)

        self.num_valid_cells_in_window = signal.convolve2d(np.ones(rd_size, dtype=float), self.mask, mode='same')

    def __call__(self, rd_matrix):
        rd_windowed_sum = signal.convolve2d(rd_matrix, self.mask, mode='same')
        rd_avg_noise_power = rd_windowed_sum / self.num_valid_cells_in_window
        threshold = self.scaling_factor * rd_avg_noise_power * 1
        hit_matrix = rd_matrix > threshold
        ind = len(hit_matrix[0]) // 2
        hit_matrix[:, ind] = False
        return hit_matrix.astype(int)

    def SNR(self, rd_matrix, snr_mask):
        rd_windowed_sum = signal.convolve2d(rd_matrix, self.mask, mode='same')
        rd_avg_noise_power = rd_windowed_sum / self.num_valid_cells_in_window
        snr = rd_matrix / (rd_avg_noise_power * 1)
        snr *= snr_mask
        # print(tabulate(np.abs(snr)))
        max_snr = snr.max()
        arg_max = np.argwhere(snr == max_snr)[0][0]
        max_snr = 10 * np.log10(max_snr)
        print(f"Max SNR of target (dB): {max_snr}\ntime: {arg_max}\n")
        return max_snr, arg_max


def get_data_by_id(f, id_):
    numChannel = 4
    id = -1

    while id != id_:
        a = np.fromfile(f, dtype=np.uint32, count=2)
        id = a[0]
        count = a[1]
        timestamp = np.fromfile(f, dtype=np.uint64, count=1)[0]
        a = np.fromfile(f, dtype=np.uint32, count=3)
        size = a[0]
        num_range_chan = a[1]
        num_doppler_chan = a[2]

        # print(id, count, timestamp, size, num_range_chan, num_doppler_chan)
        data_size = 2 * numChannel * num_range_chan * num_doppler_chan

        data_arr = np.fromfile(f, dtype=np.int16, count=data_size)
        data_arr = 1j * data_arr[1::2] + data_arr[0::2]

        data_frame = np.reshape(data_arr, (numChannel, num_range_chan, num_doppler_chan))
        np.seterr(divide='ignore')

        tmp = data_frame[0, :, :]
    return tmp


def setup_file(file_name):
    f = open(file_name, 'rb')
    num_params = np.fromfile(f, dtype=np.int32, count=1)[0]
    print(num_params)
    print()
    f.seek(num_params * 92, 1)
    return f


def remove_infs_nans(arr):
    arr[arr == float("-inf")] = 0
    arr[np.isnan(arr)] = 0
    arr[arr == float("inf")] = 1
    return arr


def to_pixels(arr):
    arr = np.round(255 * arr)
    return arr.astype(np.uint8)


def duplicates(arr):
    return [elem in arr[:i] for i, elem in enumerate(arr)]


def replace_lost(arr_y, arr_x, lost_x, lost_y):
    for i in range(len(lost_x)):
        index = np.argwhere(arr_x == lost_x[i])[0]
        arr_y[index] = lost_y[i]

    return arr_y


def flow_detector(file_name):
    snr_arr = np.array([])
    dist_arr = np.array([])
    lost_snr = np.array([])
    lost_dist = np.array([])

    frame_counter = 0
    f = setup_file(file_name)

    id = 2
    pict_shape = (240, 512)

    flow_cfar = CA_CFAR([4, 4, 2, 2], 0.0002, pict_shape)
    detect_cfar = CA_CFAR([4, 4, 2, 2], 0.00002, pict_shape)
    snr_cfar = CA_CFAR([10, 10, 3, 3], 0.01, pict_shape)

    flag = True

    while True:
        range_dopler = get_data_by_id(f, id)
        frame_counter += 1

        print(frame_counter)

        if flag:
            prev_rd = range_dopler
            prev_rd_power = remove_infs_nans(20 * np.log10(np.abs(range_dopler)))

            # if frame_counter == 495:
                # flag = False

            flag = False

        else:
            present_rd_power = remove_infs_nans(10 * np.log10(np.abs(range_dopler)))
            cv.imshow(f'id = {id}', to_pixels(prev_rd_power / prev_rd_power.max()))

            mask_flow_cfar = flow_cfar(remove_infs_nans(np.abs(prev_rd)))
            mask_flow_cfar_pres = flow_cfar(remove_infs_nans(np.abs(range_dopler)))
            mask_detect_cfar = detect_cfar(remove_infs_nans(np.abs(prev_rd)))

            cv.imshow('CFAR detection', to_pixels(mask_detect_cfar))
            cv.imshow('CFAR data prep', to_pixels(mask_flow_cfar))

            pow_rd_prev = np.abs(prev_rd)
            pow_rd_pres = np.abs(range_dopler)

            amb_func_prev = remove_infs_nans(pow_rd_prev / pow_rd_prev.max()) * mask_flow_cfar
            amb_func_pres = remove_infs_nans(pow_rd_pres / pow_rd_pres.max()) * mask_flow_cfar_pres

            flow = cv.calcOpticalFlowFarneback(amb_func_prev, amb_func_pres,
                                               None, pyr_scale=0.5, levels=1, winsize=1, iterations=1, poly_n=5,
                                               poly_sigma=1.2, flags=0)

            mag, ang = cv.cartToPolar(flow[:, :, 0], flow[:, :, 1], False)
            mag = remove_infs_nans(mag)

            if mag.max() == 0.0:
                mag = np.zeros_like(mag)
            else:
                mag = remove_infs_nans(mag / mag.max())

            flow_mask = to_pixels(mag).astype('int32')
            flow_mask[flow_mask < 150] = 0
            flow_mask[flow_mask > 0] = 1

            cv.imshow('Flow detection', to_pixels(flow_mask))

            if np.any(flow_mask != 0):
                print("FLOW DETECT")
            if np.any(mask_detect_cfar != 0):
                print("CFAR DETECT")

            prev_rd_power = present_rd_power
            prev_rd = range_dopler

        k = cv.waitKey(30) & 0xff

        if k == 27:
            k_2 = cv.waitKey(999999) & 0xff
            if k_2 == 27:
                continue
    return 0


def cfar_detector(file_name):
    frame_counter = 0
    f = setup_file(file_name)

    id = 0

    # pict_shape = (120, 512)
    #
    # detect_cfar = CA_CFAR([10, 10, 3, 3], 0.003, pict_shape)

    pict_shape = (240, 512)

    detect_cfar = CA_CFAR([4, 4, 2, 2], 0.0002, pict_shape)

    flag = True

    while True:
        range_dopler = get_data_by_id(f, id)
        frame_counter += 1
	

        # if frame_counter == 400:
            # return

        if flag:
            prev_rd_power = remove_infs_nans(20 * np.log10(np.abs(range_dopler)))

            flag = False
        else:
            present_rd_power = remove_infs_nans(20 * np.log10(np.abs(range_dopler)))
            cv.imshow(f'id = {id}', to_pixels(prev_rd_power / prev_rd_power.max()))

            mask_detect_cfar = detect_cfar(remove_infs_nans(np.abs(range_dopler)))

            cv.imshow('CFAR detection', to_pixels(mask_detect_cfar))

            prev_rd_power = present_rd_power

        k = cv.waitKey(30) & 0xff

        if k == 27:
            k_2 = cv.waitKey(999999) & 0xff
            if k_2 == 27:
                continue
    return 0


snr_info = flow_detector("F:\mavic02")
