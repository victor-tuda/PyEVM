import cv2
import numpy as np


def build_gaussian_pyramid(src, level=3):
    s = src.copy()
    pyramid = [s]
    for i in range(level):
        s = cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid


def build_laplacian_pyramid(src, levels=3):
    gaussianPyramid = build_gaussian_pyramid(src, levels)
    pyramid = []
    for i in range(levels, 0, -1):
        GE = cv2.pyrUp(gaussianPyramid[i])

        # Ensure the size of GE matches the size of the previous Gaussian pyramid level
        GE = cv2.resize(GE, (gaussianPyramid[i - 1].shape[1], gaussianPyramid[i - 1].shape[0]))

        L = cv2.subtract(gaussianPyramid[i - 1], GE)
        pyramid.append(L)
    return pyramid


def gaussian_video(video_tensor, levels=3):
    for i in range(0, video_tensor.shape[0]):
        frame = video_tensor[i]
        pyr = build_gaussian_pyramid(frame, level=levels)
        gaussian_frame = pyr[-1]
        if i == 0:
            vid_data = np.zeros((video_tensor.shape[0], gaussian_frame.shape[0], gaussian_frame.shape[1], 3))
        vid_data[i] = gaussian_frame
    return vid_data


def laplacian_video(video_tensor, levels=3):
    tensor_list = []
    for i in range(0, video_tensor.shape[0]):
        frame = video_tensor[i]
        pyr = build_laplacian_pyramid(frame, levels=levels)
        if i == 0:
            for k in range(levels):
                tensor_list.append(np.zeros((video_tensor.shape[0], pyr[k].shape[0], pyr[k].shape[1], 3)))
        for n in range(levels):
            tensor_list[n][i] = pyr[n]
    return tensor_list


def _reconstruct_from_tensor_list(filter_tensor_list):
    levels = len(filter_tensor_list)
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[-1][i]
        for n in range(levels - 2, -1, -1):
            # Upsample 'up'
            up = cv2.pyrUp(up)

            # Ensure the dimensions of 'up' match those of the current filter tensor
            filter_shape = filter_tensor_list[n][i].shape
            up = cv2.resize(up, (filter_shape[1], filter_shape[0]))

            # Resize the current filter tensor if needed to match the shape of 'up'
            filter_resized = cv2.resize(filter_tensor_list[n + 1][i], (up.shape[1], up.shape[0]))

            # Now add the upsampled 'up' to the resized filter tensor
            up = up + filter_resized

    return up
