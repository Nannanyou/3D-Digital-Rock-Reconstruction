# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import time
import re
import bisect
from collections import OrderedDict
import numpy as np
import scipy.ndimage
import scipy.misc

import config
import misc
import tfutil
import train
import dataset
import tensorflow.compat.v1 as tf
import cv2
from scipy.io import savemat
from scipy.io import savemat, loadmat
from scipy import interpolate
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib

# from proggan import from_tf_parameters
# import torch

tf.disable_v2_behavior()


# # ----------------------------------------------------------------------------
# # Generate random images or image grids using a previously trained network.
# # To run, uncomment the appropriate line in config.py and launch train.py.
#
# def pkl_to_pth(run_id, snapshot=None):
#     network_pkl = misc.locate_network_pkl(run_id, snapshot)
#     print('Loading network from "%s"...' % network_pkl)
#     G, D, Gs = misc.load_network_pkl(run_id, snapshot)
#     result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
#     PATH = os.path.join(misc.locate_result_subdir(run_id), 'carbonate.pth')
#     print(G.trainables)
#     model = from_tf_parameters(G.trainables)
#     torch.save(model, PATH)
#     return 0


# ----------------------------------------------------------------------------
# Generate random images or image grids using a previously trained network.
# To run, uncomment the appropriate line in config.py and launch train.py.

def generate_fake_images(run_id, snapshot=None, grid_size=[1, 1],
                         num_pngs=1, image_shrink=1, png_prefix=None,
                         random_seed=1000, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    if png_prefix is None:
        png_prefix = misc.get_id_string_for_network_pkl(network_pkl) + '-'
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    for png_idx in range(num_pngs):
        print('Generating png %d / %d...' % (png_idx, num_pngs))
        latents = misc.random_latents(np.prod(grid_size), Gs, random_state=random_state)
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5,
                        out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        misc.save_image_grid(images, os.path.join(result_subdir, '%s%06d.png' % (png_prefix, png_idx)), [0, 255],
                             grid_size)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()


def generate_specified_images(run_id, snapshot=None, grid_size=[1, 1],
                              image_shrink=1, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)

    z_final = loadmat('./results_new/z.mat')['z_final']
    z_initial = loadmat('./results_new/z.mat')['z_initial']
    labels = np.zeros([z_final.shape[0], 0], np.float32)
    final_images = Gs.run(z_final, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5,
                          out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
    final_images1 = np.swapaxes(final_images, axis1=0, axis2=2)  # [1024, 1, 1725, 1024]
    initial_images = Gs.run(z_initial, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5,
                            out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
    initial_images1 = np.swapaxes(initial_images, axis1=0, axis2=2)
    # for i in range(int(np.ceil(initial_images.shape[0] / minibatch_size))):
    misc.save_image_grid(initial_images1[600:601],
                         os.path.join(result_subdir, 's1-%s%06d.png' % ('initial', 600)),
                         [0, 255], grid_size)
    # for i in range(int(np.ceil(final_images.shape[0] / minibatch_size))):
    misc.save_image_grid(final_images1[600:601],
                         os.path.join(result_subdir, 's1-%s%06d.png' % ('final', 600)),
                         [0, 255], grid_size)

    final_images2 = np.swapaxes(final_images, axis1=0, axis2=3)  # [1024, 1, 1725, 1024]
    initial_images2 = np.swapaxes(initial_images, axis1=0, axis2=3)
    # for i in range(int(np.ceil(initial_images.shape[0] / minibatch_size))):
    misc.save_image_grid(initial_images2[600:601],
                         os.path.join(result_subdir, 's2-%s%06d.png' % ('initial', 600)),
                         [0, 255], grid_size)
    # for i in range(int(np.ceil(final_images.shape[0] / minibatch_size))):
    misc.save_image_grid(final_images2[600:601],
                         os.path.join(result_subdir, 's2-%s%06d.png' % ('final', 600)),
                         [0, 255], grid_size)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()


def interpolation_between_z(run_id, snapshot=None, grid_size=[1, 1],
                            image_shrink=1, minibatch_size=8, window_size=8, start=3, end=1720):
    import matplotlib.pyplot as plt

    network_pkl = misc.locate_network_pkl(run_id, snapshot)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)

    z_final = loadmat('./results_new/z_%d_%d_%d.mat' % (start, window_size, run_id))['z_final']

    # xp = np.arange(start, 1725, window_size)
    # xp = np.concatenate((np.arange(0, 512, 9), np.arange(512, 701, 9), np.arange(701, 1725, 9)))
    xp = np.arange(start, end, window_size)

    # z_slinear = np.zeros((end, z_final.shape[1]), dtype=np.float32)
    # for i in range(z_final.shape[1]):
    #     f = interpolate.UnivariateSpline(xp, z_final[:, i], k=1, ext=0, s=0)
    #     z_slinear[:, i] = f(np.arange(0, end, 1))
    f = interpolate.interp1d(xp, z_final, axis=0, kind='linear', fill_value='extrapolate')
    z_slinear = f(np.arange(0, end, 1))

    # plt.figure(figsize=(9, 6))
    # plt.plot(np.arange(end), z_slinear[:, 1], label='$z_{linear}[1]$')
    # plt.plot(xp, z_final[:, 1], 'r.', label='data')
    # plt.legend()
    # plt.ylim([-6, 6])
    # plt.title('Linear interpolation of z[1]')
    # plt.savefig(result_subdir + '/slinear.png')
    # plt.close()

    labels = np.zeros([z_slinear.shape[0], 0], np.float32)

    for k, v in {
        'z_linear': z_slinear
    }.items():
        print('----- %s -----' % k)
        images = Gs.run(v, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5,
                        out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)  # (1725, 1, 1024, 1024)

        ths1 = []
        ths2 = []
        for img in images:
            pixel_set = np.expand_dims(img.flatten(), axis=-1)
            kmeans = KMeans(n_clusters=3, random_state=0, verbose=0).fit(pixel_set)
            cluster1 = np.squeeze(pixel_set[kmeans.labels_ == 0])
            # print('Class 1')
            # print(np.max(cluster1))
            # print(np.min(cluster1))
            cluster2 = np.squeeze(pixel_set[kmeans.labels_ == 1])
            # print('Class 2')
            # print(np.max(cluster2))
            # print(np.min(cluster2))
            cluster3 = np.squeeze(pixel_set[kmeans.labels_ == 2])
            # print('Class 3')
            # print(np.max(cluster3))
            # print(np.min(cluster3))
            max = [np.max(cluster1), np.max(cluster2), np.max(cluster3)]
            max.sort()
            ths1.append(max[0])
            ths2.append(max[1])
        print(np.bincount(ths1, minlength=256))
        print(np.bincount(ths2, minlength=256))
        print(np.mean(ths1))
        print(np.mean(ths2))
        ths1 = stats.mode(ths1)[0][0]
        ths2 = stats.mode(ths2)[0][0]
        print('Thres1: ' + str(ths1))
        print('Thres2: ' + str(ths2))


    #     for i, img in enumerate(images):
    #         misc.save_image_grid(img,
    #                              os.path.join(result_subdir, 'fakes%04d.jpeg' % i),
    #                              [0, 255], grid_size)
    #
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()


def generate_images_from_manipulated_noise(run_id, snapshot=None, grid_size=[1, 1],
                                           image_shrink=1, minibatch_size=8, window_size=8, start=3, end=1720):
    import matplotlib.pyplot as plt

    network_pkl = misc.locate_network_pkl(run_id, snapshot)

    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)
    real_imgs, _ = training_set.get_minibatch_np(minibatch_size=end)
    real_imgs = real_imgs / 127.5 - 1

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)

    z_final = loadmat('./results_new/z_%d_%d_%d.mat' % (start, window_size, run_id))['z_final']
    # z_initial = loadmat('./results_new/z_%d_%d_%d.mat' % (start, window_size, run_id))['z_initial']
    # z_final = loadmat('./results_new/z108.mat')['z_final']
    # z_initial = loadmat('./results_new/z108.mat')['z_initial']

    # plt.figure()
    # n, bins, _ = plt.hist(z_final.flatten() - z_initial.flatten(), bins=np.arange(-10, 10, 0.01),
    #                       label='z_final - z_initial', density=True, facecolor='g', alpha=0.75)
    # plt.savefig(result_subdir + '/noise.png')
    # plt.close()
    # xn = (bins[0:-1] + bins[1:]) / 2

    # xp = np.arange(start, 1725, window_size)
    # xp = np.concatenate((np.arange(0, 512, 9), np.arange(512, 701, 9), np.arange(701, 1725, 9)))
    xp = np.arange(start, end, window_size)

    # nearest neighbour interp
    f = interpolate.interp1d(xp, z_final, kind='nearest', axis=0, fill_value="extrapolate")
    z_nn = f(np.arange(0, end, 1))
    # z_nn_normed = z_nn / np.sqrt(np.mean(np.square(z_nn), axis=1, keepdims=True))
    # z_nn_gsmooth = scipy.ndimage.gaussian_filter(z_nn, [window_size, 0], mode='nearest')
    # z_nn_usmooth = scipy.ndimage.uniform_filter(z_nn, [window_size, 0], mode='nearest')
    # z_nn_smooth_normed = z_nn_smooth / np.sqrt(np.mean(np.square(z_nn_smooth), axis=1, keepdims=True))
    # noise = np.random.choice(xn, size=z_nn.shape, replace=True, p=n * 0.01)
    # z_nn_noise = z_nn + noise
    # z_nn_noise_normed = z_nn_noise / np.sqrt(np.mean(np.square(z_nn_noise), axis=1, keepdims=True))
    # z_nn_noise_gsmooth = scipy.ndimage.gaussian_filter(z_nn_noise, [window_size, 0], mode='nearest')
    # z_nn_noise_usmooth = scipy.ndimage.uniform_filter(z_nn_noise, [window_size, 0], mode='nearest')
    # z_nn_noise_smooth_normed = z_nn_noise_smooth / np.sqrt(np.mean(np.square(z_nn_noise_smooth), axis=1, keepdims=True))
    # z_nn_smooth_noise = z_nn_smooth + noise
    # z_nn_smooth_noise_normed = z_nn_smooth_noise / np.sqrt(
    #     np.mean(np.square(z_nn_smooth_noise), axis=1, keepdims=True))

    # z_initial_noise = np.repeat(z_initial, repeats=16, axis=0)[0:1725] + noise
    # z_initial_noise_smooth = scipy.ndimage.gaussian_filter(z_initial_noise, [16, 0], mode='nearest')

    # # linear interp
    # f = interpolate.interp1d(np.arange(7, 1725, 16), z_final, kind='linear', axis=0,
    #                          fill_value='extrapolate')
    # z_linear = f(np.arange(0, 1725, 1))
    #
    # # z_linear_normed = z_linear / np.sqrt(np.mean(np.square(z_linear), axis=1, keepdims=True))
    # z_linear_smooth = scipy.ndimage.gaussian_filter(z_linear, [16, 0], mode='nearest')
    # # z_linear_smooth_normed = z_linear_smooth / np.sqrt(np.mean(np.square(z_linear_smooth), axis=1, keepdims=True))
    # z_linear_noise = z_linear + noise
    # # z_linear_noise_normed = z_linear_noise / np.sqrt(np.mean(np.square(z_linear_noise), axis=1, keepdims=True))
    # z_linear_noise_smooth = scipy.ndimage.gaussian_filter(z_linear_noise, [16, 0], mode='nearest')
    # # z_linear_noise_smooth_normed = z_linear_noise_smooth / np.sqrt(
    # #     np.mean(np.square(z_linear_noise_smooth), axis=1, keepdims=True))
    # # z_linear_smooth_noise = z_linear_smooth + noise
    # # z_linear_smooth_noise_normed = z_linear_smooth_noise / np.sqrt(
    # #     np.mean(np.square(z_linear_smooth_noise), axis=1, keepdims=True))
    # # z_linear_smooth_noise_smooth = scipy.ndimage.gaussian_filter(z_linear_smooth_noise, [16, 0], mode='nearest')
    # # z_nn_smooth_noise_smooth = scipy.ndimage.gaussian_filter(z_nn_smooth_noise, [16, 0], mode='nearest')
    #
    # # quadratic spline interp
    # f = interpolate.interp1d(np.arange(7, 1725, 16), z_final, kind='quadratic', axis=0,
    #                          fill_value='extrapolate')
    # z_quad = f(np.arange(0, 1725, 1))
    # z_quad_smooth = scipy.ndimage.gaussian_filter(z_quad, [16, 0], mode='nearest')
    # z_quad_noise = z_quad + noise
    # z_quad_noise_smooth = scipy.ndimage.gaussian_filter(z_quad_noise, [16, 0], mode='nearest')
    # # cubic spline interp
    # f = interpolate.interp1d(np.arange(7, 1725, 16), z_final, kind='cubic', axis=0,
    #                          fill_value='extrapolate')
    # z_cubic = f(np.arange(0, 1725, 1))
    # z_cubic_smooth = scipy.ndimage.gaussian_filter(z_cubic, [16, 0], mode='nearest')
    # z_cubic_noise = z_cubic + noise
    # z_cubic_noise_smooth = scipy.ndimage.gaussian_filter(z_cubic_noise, [16, 0], mode='nearest')
    # spline interp (1)
    z_slinear = np.zeros_like(z_nn)
    for i in range(z_final.shape[1]):
        f = interpolate.UnivariateSpline(xp, z_final[:, i], k=1, ext=0, s=0)
        z_slinear[:, i] = f(np.arange(0, end, 1))
    # z_slinear_gsmooth = scipy.ndimage.gaussian_filter(z_slinear, [window_size, 0], mode='nearest')
    # z_slinear_usmooth = scipy.ndimage.uniform_filter(z_slinear, [window_size, 0], mode='nearest')
    # z_slinear_noise = z_slinear + noise
    # z_slinear_noise_gsmooth = scipy.ndimage.gaussian_filter(z_slinear_noise, [window_size, 0], mode='nearest')
    # z_slinear_noise_usmooth = scipy.ndimage.uniform_filter(z_slinear_noise, [window_size, 0], mode='nearest')

    # # spline interp (2)
    # z_quad = np.zeros_like(z_nn)
    # for i in range(z_final.shape[1]):
    #     f = interpolate.UnivariateSpline(xp, z_final[:, i], k=2, ext=0, s=0)
    #     z_quad[:, i] = f(np.arange(0, 1725, 1))
    # z_quad_gsmooth = scipy.ndimage.gaussian_filter(z_quad, [window_size, 0], mode='nearest')
    # z_quad_usmooth = scipy.ndimage.uniform_filter(z_quad, [window_size, 0], mode='nearest')
    # z_quad_noise = z_quad + noise
    # z_quad_noise_gsmooth = scipy.ndimage.gaussian_filter(z_quad_noise, [window_size, 0], mode='nearest')
    # z_quad_noise_usmooth = scipy.ndimage.uniform_filter(z_quad_noise, [window_size, 0], mode='nearest')
    #
    # # spline interp (3)
    # z_cubic = np.zeros_like(z_nn)
    # for i in range(z_final.shape[1]):
    #     f = interpolate.UnivariateSpline(xp, z_final[:, i], k=3, ext=0, s=0)
    #     z_cubic[:, i] = f(np.arange(0, 1725, 1))
    # z_cubic_gsmooth = scipy.ndimage.gaussian_filter(z_cubic, [window_size, 0], mode='nearest')
    # z_cubic_usmooth = scipy.ndimage.uniform_filter(z_cubic, [window_size, 0], mode='nearest')
    # z_cubic_noise = z_cubic + noise
    # z_cubic_noise_gsmooth = scipy.ndimage.gaussian_filter(z_cubic_noise, [window_size, 0], mode='nearest')
    # z_cubic_noise_usmooth = scipy.ndimage.uniform_filter(z_cubic_noise, [window_size, 0], mode='nearest')
    #
    # # spline interp (4)
    # z_quartic = np.zeros_like(z_cubic)
    # for i in range(z_final.shape[1]):
    #     f = interpolate.UnivariateSpline(xp, z_final[:, i], k=4, ext=0, s=0)
    #     z_quartic[:, i] = f(np.arange(0, 1725, 1))
    # z_quartic_gsmooth = scipy.ndimage.gaussian_filter(z_quartic, [window_size, 0], mode='nearest')
    # z_quartic_usmooth = scipy.ndimage.uniform_filter(z_quartic, [window_size, 0], mode='nearest')
    # z_quartic_noise = z_quartic + noise
    # z_quartic_noise_gsmooth = scipy.ndimage.gaussian_filter(z_quartic_noise, [window_size, 0], mode='nearest')
    # z_quartic_noise_usmooth = scipy.ndimage.uniform_filter(z_quartic_noise, [window_size, 0], mode='nearest')
    #
    # # spline interp (5)
    # z_quintic = np.zeros_like(z_cubic)
    # for i in range(z_final.shape[1]):
    #     f = interpolate.UnivariateSpline(xp, z_final[:, i], k=5, ext=0, s=0)
    #     z_quintic[:, i] = f(np.arange(0, 1725, 1))
    # z_quintic_gsmooth = scipy.ndimage.gaussian_filter(z_quintic, [window_size, 0], mode='nearest')
    # z_quintic_usmooth = scipy.ndimage.uniform_filter(z_quintic, [window_size, 0], mode='nearest')
    # z_quintic_noise = z_quintic + noise
    # z_quintic_noise_gsmooth = scipy.ndimage.gaussian_filter(z_quintic_noise, [window_size, 0], mode='nearest')
    # z_quintic_noise_usmooth = scipy.ndimage.uniform_filter(z_quintic_noise, [window_size, 0], mode='nearest')

    # # barycentric interp
    # z_barycentric = np.zeros_like(z_cubic)
    # for i in range(z_final.shape[1]):
    #     z_barycentric[:, i] = interpolate.barycentric_interpolate(np.arange(start, 1725, window_size), z_final[:, i],
    #                                                               np.arange(0, 1725, 1), axis=0)
    # z_barycentric_gsmooth = scipy.ndimage.gaussian_filter(z_barycentric, [window_size, 0], mode='nearest')
    # z_barycentric_usmooth = scipy.ndimage.uniform_filter(z_barycentric, [window_size, 0], mode='nearest')
    # z_barycentric_noise = z_barycentric + noise
    # z_barycentric_noise_gsmooth = scipy.ndimage.gaussian_filter(z_barycentric_noise, [window_size, 0], mode='nearest')
    # z_barycentric_noise_usmooth = scipy.ndimage.uniform_filter(z_barycentric_noise, [window_size, 0], mode='nearest')

    # # Krogh interp
    # z_krogh = np.zeros_like(z_cubic)
    # for i in range(z_final.shape[1]):
    #     f = interpolate.KroghInterpolator(np.arange(start, 1725, window_size), z_final[:, i], axis=0)
    #     z_krogh[:, i] = f(np.arange(0, 1725, 1))
    # z_krogh_smooth = scipy.ndimage.gaussian_filter(z_krogh, [window_size, 0], mode='nearest')
    # z_krogh_noise = z_krogh + noise
    # z_krogh_noise_smooth = scipy.ndimage.gaussian_filter(z_krogh_noise, [window_size, 0], mode='nearest')

    # # pchip interp
    # z_pchip = interpolate.pchip_interpolate(xp, z_final, np.arange(0, 1725, 1), axis=0)
    # z_pchip_gsmooth = scipy.ndimage.gaussian_filter(z_pchip, [window_size, 0], mode='nearest')
    # z_pchip_usmooth = scipy.ndimage.uniform_filter(z_pchip, [window_size, 0], mode='nearest')
    # z_pchip_noise = z_pchip + noise
    # z_pchip_noise_gsmooth = scipy.ndimage.gaussian_filter(z_pchip_noise, [window_size, 0], mode='nearest')
    # z_pchip_noise_usmooth = scipy.ndimage.uniform_filter(z_pchip_noise, [window_size, 0], mode='nearest')
    #
    # # Akima1DInterpolator
    # f = interpolate.Akima1DInterpolator(xp, z_final, axis=0)
    # z_aki = f(np.arange(0, 1725, 1))
    # z_aki_gsmooth = scipy.ndimage.gaussian_filter(z_aki, [window_size, 0], mode='nearest')
    # z_aki_usmooth = scipy.ndimage.uniform_filter(z_aki, [window_size, 0], mode='nearest')
    # z_aki_noise = z_aki + noise
    # z_aki_noise_gsmooth = scipy.ndimage.gaussian_filter(z_aki_noise, [window_size, 0], mode='nearest')
    # z_aki_noise_usmooth = scipy.ndimage.uniform_filter(z_aki_noise, [window_size, 0], mode='nearest')

    # plt.figure(figsize=(9, 6))
    # plt.plot(np.arange(end), z_slinear[:, 1], label='$z_{linear}[1]$')
    # plt.plot(np.arange(end), z_slinear_usmooth[:, 1], label='$z_{linear\\_unif-smooth}[1]$')
    # plt.plot(np.arange(end), z_slinear_gsmooth[:, 1], label='$z_{linear\\_gau-smooth}[1]$')
    # plt.plot(xp, z_final[:, 1], 'r.', label='data')
    # plt.legend()
    # plt.ylim([-6, 6])
    # plt.title('Linear interpolation of z[1]')
    # plt.savefig(result_subdir + '/slinear.png')
    # plt.close()
    #
    # plt.figure(figsize=(9, 6))
    # plt.plot(np.arange(end), z_nn[:, 1], label='$z_{NN}[1]$')
    # # plt.plot(np.arange(end), z_nn_usmooth[:, 1], label='$z_{NN\\_unif-smooth}[1]$')
    # plt.plot(np.arange(end), z_nn_gsmooth[:, 1], label='$z_{NN\\_gau-smooth}[1]$')
    # plt.plot(xp, z_final[:, 1], 'r.', label='data')
    # plt.legend()
    # plt.ylim([-6, 6])
    # plt.title('NN interpolation of z[1]')
    # plt.savefig(result_subdir + '/nn.png')
    # plt.close()

    # plt.figure(figsize=(9, 6))
    # plt.plot(np.arange(1725), z_quartic[:, 1], label='$z_{quartic-spline}[1]$')
    # plt.plot(np.arange(1725), z_quartic_usmooth[:, 1], label='$z_{quartic-spline\\_unif-smooth}[1]$')
    # plt.plot(np.arange(1725), z_quartic_gsmooth[:, 1], label='$z_{quartic-spline\\_gau-smooth}[1]$')
    # plt.plot(xp, z_final[:, 1], 'r.', label='data')
    # plt.legend()
    # plt.ylim([-6, 6])
    # plt.title('4th order spline interpolation of z[1]')
    # plt.savefig(result_subdir + '/quartic.png')
    # plt.close()
    #
    # plt.figure(figsize=(9, 6))
    # plt.plot(np.arange(1725), z_quintic[:, 1], label='$z_{quintic-spline}[1]$')
    # plt.plot(np.arange(1725), z_quintic_usmooth[:, 1], label='$z_{quartic-spline\\_unif-smooth}[1]$')
    # plt.plot(np.arange(1725), z_quintic_gsmooth[:, 1], label='$z_{quartic-spline\\_gau-smooth}[1]$')
    # plt.plot(xp, z_final[:, 1], 'r.', label='data')
    # plt.legend()
    # plt.ylim([-6, 6])
    # plt.title('5th order spline interpolation of z[1]')
    # plt.savefig(result_subdir + '/quintic.png')
    # plt.close()
    #
    # plt.figure(figsize=(9, 6))
    # plt.plot(np.arange(1725), z_quad[:, 1], label='$z_{quad-spline}[1]$')
    # plt.plot(np.arange(1725), z_quad_usmooth[:, 1], label='$z_{quad-spline\\_unif-smooth}[1]$')
    # plt.plot(np.arange(1725), z_quad_gsmooth[:, 1], label='$z_{quad-spline\\_gau-smooth}[1]$')
    # plt.plot(xp, z_final[:, 1], 'r.', label='data')
    # plt.legend()
    # plt.ylim([-6, 6])
    # plt.title('2nd order spline interpolation of z[1]')
    # plt.savefig(result_subdir + '/quad.png')
    # plt.close()
    #
    # plt.figure(figsize=(9, 6))
    # plt.plot(np.arange(1725), z_cubic[:, 1], label='$z_{cubic-spline}[1]$')
    # plt.plot(np.arange(1725), z_cubic_usmooth[:, 1], label='$z_{cubic-spline\\_unif-smooth}[1]$')
    # plt.plot(np.arange(1725), z_cubic_gsmooth[:, 1], label='$z_{cubic-spline\\_gau-smooth}[1]$')
    # plt.plot(xp, z_final[:, 1], 'r.', label='data')
    # plt.legend()
    # plt.ylim([-6, 6])
    # plt.title('3rd order spline interpolation of z[1]')
    # plt.savefig(result_subdir + '/cubic.png')
    # plt.close()

    # plt.figure(figsize=(9, 9))
    # plt.plot(np.arange(1725), z_barycentric[:, 1], label='barycentric')
    # plt.plot(np.arange(1725), z_barycentric_smooth[:, 1], label='barycentric_smo0th')
    # plt.plot(np.arange(7, 1725, 16), z_final[:, 1], 'r.')
    # plt.legend()
    # plt.savefig(result_subdir + '/barycentric.png')
    # plt.close()
    #
    # plt.figure(figsize=(9, 9))
    # plt.plot(np.arange(1725), z_krogh[:, 1], label='Krogh')
    # plt.plot(np.arange(1725), z_krogh_smooth[:, 1], label='Krogh_smooth')
    # plt.plot(np.arange(7, 1725, 16), z_final[:, 1], 'r.')
    # plt.legend()
    # plt.savefig(result_subdir + '/krogh.png')
    # plt.close()

    # plt.figure(figsize=(9, 6))
    # plt.plot(np.arange(1725), z_pchip[:, 1], label='$z_{pchip}[1]$')
    # plt.plot(np.arange(1725), z_pchip_usmooth[:, 1], label='$z_{pchip\\_unif-smooth}[1]$')
    # plt.plot(np.arange(1725), z_pchip_gsmooth[:, 1], label='$z_{pchip\\_gau-smooth}[1]$')
    # plt.plot(xp, z_final[:, 1], 'r.', label='data')
    # plt.legend()
    # plt.ylim([-6, 6])
    # plt.title('PCHIP interpolation of z[1]')
    # plt.savefig(result_subdir + '/pchip.png')
    # plt.close()

    # plt.figure(figsize=(9, 9))
    # plt.plot(np.arange(1725), z_aki[:, 1], label='aki')
    # plt.plot(np.arange(1725), z_aki_smooth[:, 1], label='aki_smooth')
    # plt.plot(np.arange(7, 1725, 16), z_final[:, 1], 'r.')
    # plt.legend()
    # plt.savefig(result_subdir + '/aki.png')
    # plt.close()

    # plt.figure(figsize=(9, 9))
    # plt.hist(noise.flatten(), bins=np.arange(-8, 8, 0.01), label='noise', density=True, alpha=0.75)
    # plt.legend()
    # plt.savefig(result_subdir + '/noise_sampled.png')
    # plt.close()

    # plt.figure(figsize=(9, 6))
    # for i, (k, v) in enumerate({'z_nn_normed': z_nn_normed, 'z_nn_smooth_normed': z_nn_smooth_normed,
    #                             'z_nn_noise_normed': z_nn_noise_normed,
    #                             'z_nn_smooth_noise_normed': z_nn_smooth_noise_normed,
    #                             'z_nn_noise_smooth_normed': z_nn_noise_smooth_normed}.items()):
    #     plt.subplot(2, 3, i + 1)
    #     plt.hist(v.flatten(), bins=50, density=True, alpha=0.75)
    #     plt.xlim([-10, 10])
    #     plt.ylim([0, 1])
    #     plt.title(k)
    # plt.savefig(result_subdir + '/z_nn_hist_normed.png')
    #
    # plt.figure(figsize=(9, 6))
    # for i, (k, v) in enumerate({'z_linear_normed': z_linear_normed, 'z_linear_smooth_normed': z_linear_smooth_normed,
    #                             'z_linear_noise_normed': z_linear_noise_normed,
    #                             'z_linear_smooth_noise_normed': z_linear_smooth_noise_normed,
    #                             'z_linear_noise_smooth_normed': z_linear_noise_smooth_normed}.items()):
    #     plt.subplot(2, 3, i + 1)
    #     plt.hist(v.flatten(), bins=50, density=True, alpha=0.75)
    #     plt.xlim([-10, 10])
    #     plt.ylim([0, 1])
    #     plt.title(k)
    # plt.savefig(result_subdir + '/z_linear_hist_normed.png')

    # plt.figure(figsize=(9, 6))
    # for i, (k, v) in enumerate({'z_nn': z_nn,
    #                             'z_nn_gaussian-smooth': z_nn_gsmooth,
    #                             'z_nn_uniform-smooth': z_nn_usmooth,
    #                             'z_nn_noise': z_nn_noise,
    #                             'z_nn_noise_gaussian-smooth': z_nn_noise_gsmooth,
    #                             'z_nn_noise_uniform-smooth': z_nn_noise_usmooth}.items()):
    #     plt.subplot(2, 3, i + 1)
    #     plt.hist(v.flatten(), bins=50, density=True, alpha=0.75)
    #     plt.xlim([-10, 10])
    #     plt.ylim([0, 1])
    #     plt.title(k)
    # plt.savefig(result_subdir + '/z_nn_hist.png')
    # plt.close()
    #
    # plt.figure(figsize=(9, 6))
    # for i, (k, v) in enumerate({'z_linear': z_slinear,
    #                             'z_linear_gaussian-smooth': z_slinear_gsmooth,
    #                             'z_linear_uniform-smooth': z_slinear_usmooth,
    #                             'z_linear_noise': z_slinear_noise,
    #                             'z_linear_noise_gaussian-smooth': z_slinear_noise_gsmooth,
    #                             'z_linear_noise_uniform-smooth': z_slinear_noise_usmooth}.items()):
    #     plt.subplot(2, 3, i + 1)
    #     plt.hist(v.flatten(), bins=50, density=True, alpha=0.75)
    #     plt.xlim([-10, 10])
    #     plt.ylim([0, 1])
    #     plt.title(k)
    # plt.savefig(result_subdir + '/z_slinear_hist.png')
    # plt.close()

    # plt.figure(figsize=(9, 6))
    # for i, (k, v) in enumerate({'z_quintic': z_quintic,
    #                             'z_quintic_gaussian-smooth': z_quintic_gsmooth,
    #                             'z_quintic_uniform-smooth': z_quintic_usmooth,
    #                             'z_quintic_noise': z_quintic_noise,
    #                             'z_quintic_noise_gaussian-smooth': z_quintic_noise_gsmooth,
    #                             'z_quintic_noise_uniform-smooth': z_quintic_noise_usmooth}.items()):
    #     plt.subplot(2, 3, i + 1)
    #     plt.hist(v.flatten(), bins=50, density=True, alpha=0.75)
    #     plt.xlim([-10, 10])
    #     plt.ylim([0, 1])
    #     plt.title(k)
    # plt.savefig(result_subdir + '/z_quintic_hist.png')
    # plt.close()
    #
    # plt.figure(figsize=(9, 6))
    # for i, (k, v) in enumerate({'z_quartic': z_quartic,
    #                             'z_quartic_gaussian-smooth': z_quartic_gsmooth,
    #                             'z_quartic_uniform-smooth': z_quartic_usmooth,
    #                             'z_quartic_noise': z_quartic_noise,
    #                             'z_quartic_noise_gaussian-smooth': z_quartic_noise_gsmooth,
    #                             'z_quartic_noise_uniform-smooth': z_quartic_noise_usmooth}.items()):
    #     plt.subplot(2, 3, i + 1)
    #     plt.hist(v.flatten(), bins=50, density=True, alpha=0.75)
    #     plt.xlim([-10, 10])
    #     plt.ylim([0, 1])
    #     plt.title(k)
    # plt.savefig(result_subdir + '/z_quartic_hist.png')
    # plt.close()
    #
    # plt.figure(figsize=(9, 6))
    # for i, (k, v) in enumerate({'z_quad': z_quad,
    #                             'z_quad_gaussian-smooth': z_quad_gsmooth,
    #                             'z_quad_uniform-smooth': z_quad_usmooth,
    #                             'z_quad_noise': z_quad_noise,
    #                             'z_quad_noise_gaussian-smooth': z_quad_noise_gsmooth,
    #                             'z_quad_noise_uniform-smooth': z_quad_noise_usmooth}.items()):
    #     plt.subplot(2, 3, i + 1)
    #     plt.hist(v.flatten(), bins=50, density=True, alpha=0.75)
    #     plt.xlim([-10, 10])
    #     plt.ylim([0, 1])
    #     plt.title(k)
    # plt.savefig(result_subdir + '/z_quad_hist.png')
    # plt.close()
    #
    # plt.figure(figsize=(9, 6))
    # for i, (k, v) in enumerate({'z_cubic': z_cubic,
    #                             'z_cubic_gaussian-smooth': z_cubic_gsmooth,
    #                             'z_cubic_uniform-smooth': z_cubic_usmooth,
    #                             'z_cubic_noise': z_cubic_noise,
    #                             'z_cubic_noise_gaussian-smooth': z_cubic_noise_gsmooth,
    #                             'z_cubic_noise_uniform-smooth': z_cubic_noise_usmooth}.items()):
    #     plt.subplot(2, 3, i + 1)
    #     plt.hist(v.flatten(), bins=50, density=True, alpha=0.75)
    #     plt.xlim([-10, 10])
    #     plt.ylim([0, 1])
    #     plt.title(k)
    # plt.savefig(result_subdir + '/z_cubic_hist.png')
    # plt.close()

    # plt.figure(figsize=(9, 6))
    # for i, (k, v) in enumerate({'z_barycentric': z_barycentric,
    #                             'z_barycentric_smooth': z_barycentric_smooth,
    #                             'z_barycentric_noise': z_barycentric_noise,
    #                             'z_barycentric_noise_smooth': z_barycentric_noise_smooth}.items()):
    #     plt.subplot(2, 3, i + 1)
    #     plt.hist(v.flatten(), bins=50, density=True, alpha=0.75)
    #     plt.xlim([-10, 10])
    #     plt.ylim([0, 1])
    #     plt.title(k)
    # plt.savefig(result_subdir + '/z_barycentric_hist.png')
    # plt.close()
    #
    # plt.figure(figsize=(9, 6))
    # for i, (k, v) in enumerate({'z_krogh': z_krogh,
    #                             'z_krogh_smooth': z_krogh_smooth,
    #                             'z_krogh_noise': z_krogh_noise,
    #                             'z_krogh_noise_smooth': z_krogh_noise_smooth}.items()):
    #     plt.subplot(2, 3, i + 1)
    #     plt.hist(v.flatten(), bins=50, density=True, alpha=0.75)
    #     plt.xlim([-10, 10])
    #     plt.ylim([0, 1])
    #     plt.title(k)
    # plt.savefig(result_subdir + '/z_krogh_hist.png')
    # plt.close()

    # plt.figure(figsize=(9, 6))
    # for i, (k, v) in enumerate({'z_pchip': z_pchip,
    #                             'z_pchip_gaussian-smooth': z_pchip_gsmooth,
    #                             'z_pchip_uniform-smooth': z_pchip_usmooth,
    #                             'z_pchip_noise': z_pchip_noise,
    #                             'z_pchip_noise_gaussian-smooth': z_pchip_noise_gsmooth,
    #                             'z_pchip_noise_uniform-smooth': z_pchip_noise_usmooth}.items()):
    #     plt.subplot(2, 3, i + 1)
    #     plt.hist(v.flatten(), bins=50, density=True, alpha=0.75)
    #     plt.xlim([-10, 10])
    #     plt.ylim([0, 1])
    #     plt.title(k)
    # plt.savefig(result_subdir + '/z_pchip_hist.png')
    # plt.close()

    # plt.figure(figsize=(9, 6))
    # for i, (k, v) in enumerate({'z_aki': z_aki,
    #                             'z_aki_smooth': z_aki_smooth,
    #                             'z_aki_noise': z_aki_noise,
    #                             'z_aki_noise_smooth': z_aki_noise_smooth}.items()):
    #     plt.subplot(2, 3, i + 1)
    #     plt.hist(v.flatten(), bins=50, density=True, alpha=0.75)
    #     plt.xlim([-10, 10])
    #     plt.ylim([0, 1])
    #     plt.title(k)
    # plt.savefig(result_subdir + '/z_aki_hist.png')
    # plt.close()

    misc.save_image_grid(np.swapaxes(real_imgs, axis1=0, axis2=2)[600:601, :, 1530:1720],
                         os.path.join(result_subdir, 's1-test-real%06d.png' % 600),
                         [-1, 1], grid_size)
    misc.save_image_grid(np.swapaxes(real_imgs, axis1=0, axis2=3)[600:601, :, :, 1530:1720],
                         os.path.join(result_subdir, 's2-test-real%06d.png' % 600),
                         [-1, 1], grid_size)

    labels = np.zeros([z_nn.shape[0], 0], np.float32)
    # ths1_real = []
    # ths2_real = []
    # for img in ((1 + real_imgs) * 127.5):
    #     pixel_set = np.expand_dims(img.flatten(), axis=-1)
    #     kmeans = KMeans(n_clusters=2, random_state=0, verbose=0).fit(pixel_set)
    #     cluster1 = np.squeeze(pixel_set[kmeans.labels_ == 0])
    #     # print('Class 1')
    #     # print(np.max(cluster1))
    #     # print(np.min(cluster1))
    #     cluster2 = np.squeeze(pixel_set[kmeans.labels_ == 1])
    #     # print('Class 2')
    #     # print(np.max(cluster2))
    #     # print(np.min(cluster2))
    #     # cluster3 = np.squeeze(pixel_set[kmeans.labels_ == 2])
    #     # print('Class 3')
    #     # print(np.max(cluster3))
    #     # print(np.min(cluster3))
    #     max = [np.max(cluster1), np.max(cluster2)]
    #     max.sort()
    #     ths1_real.append(max[0])
    #     ths2_real.append(max[1])
    # print(np.bincount(ths1_real, minlength=256))
    # print(np.bincount(ths2_real, minlength=256))
    # print(np.mean(ths1_real))
    # print(np.mean(ths2_real))
    # ths1_real = stats.mode(ths1_real)[0][0]
    # ths2_real = stats.mode(ths2_real)[0][0]
    # print('Thres1: ' + str(ths1_real))
    # print('Thres2: ' + str(ths2_real))
    ths1_real = 96
    ths2_real = 143

    seg_real = np.zeros_like(real_imgs, dtype=np.float32)
    seg_real[(1 + real_imgs) * 127.5 <= ths1_real] = 0
    seg_real[np.logical_and((1 + real_imgs) * 127.5 >= ths1_real + 1,
                            (1 + real_imgs) * 127.5 <= ths2_real)] = 127.5
    seg_real[(1 + real_imgs) * 127.5 >= ths2_real + 1] = 255

    misc.save_image_grid(np.swapaxes(seg_real, axis1=0, axis2=2)[600:601],
                         os.path.join(result_subdir, 's1-seg-real%06d.png' % 600),
                         [0, 255], grid_size)
    misc.save_image_grid(np.swapaxes(seg_real, axis1=0, axis2=3)[600:601],
                         os.path.join(result_subdir, 's2-seg-real%06d.png' % 600),
                         [0, 255], grid_size)
    real1 = np.swapaxes(real_imgs, axis1=0, axis2=2)  # [1024, 1, 1725, 1024]
    real2 = np.swapaxes(real_imgs, axis1=0, axis2=3)  # [1024, 1, 1024, 1725]

    misc.save_image_grid(real1[600:601],
                         os.path.join(result_subdir, 's1-%s%06d.png' % ('real', 600)),
                         [-1, 1], grid_size)
    misc.save_image_grid(real2[600:601],
                         os.path.join(result_subdir, 's2-%s%06d.png' % ('real', 600)),
                         [-1, 1], grid_size)

    for k, v in {
        'z_nn': z_nn,
        # 'z_nn_gaussian-smooth': z_nn_gsmooth,
        # 'z_nn_uniform-smooth': z_nn_usmooth,
        # 'z_nn_noise': z_nn_noise,
        # 'z_nn_noise_gaussian-smooth': z_nn_noise_gsmooth,
        # 'z_nn_noise_uniform-smooth': z_nn_noise_usmooth,
        'z_linear': z_slinear,
        # 'z_linear_gaussian-smooth': z_slinear_gsmooth,
        # 'z_linear_uniform-smooth': z_slinear_usmooth,
        # 'z_linear_noise': z_slinear_noise,
        # 'z_linear_noise_gaussian-smooth': z_slinear_noise_gsmooth,
        # 'z_linear_noise_uniform-smooth': z_slinear_noise_usmooth,
        # 'z_quartic': z_quartic,
        # 'z_quartic_gaussian-smooth': z_quartic_gsmooth,
        # 'z_quartic_uniform-smooth': z_quartic_usmooth,
        # 'z_quartic_noise': z_quartic_noise,
        # 'z_quartic_noise_gaussian-smooth': z_quartic_noise_gsmooth,
        # 'z_quartic_noise_uniform-smooth': z_quartic_noise_usmooth,
        # 'z_quintic': z_quintic,
        # 'z_quintic_gaussian-smooth': z_quintic_gsmooth,
        # 'z_quintic_uniform-smooth': z_quintic_usmooth,
        # 'z_quintic_noise': z_quintic_noise,
        # 'z_quintic_noise_gaussian-smooth': z_quintic_noise_gsmooth,
        # 'z_quintic_noise_uniform-smooth': z_quintic_noise_usmooth,
        # 'z_quad': z_quad,
        # 'z_quad_gaussian-smooth': z_quad_gsmooth,
        # 'z_quad_uniform-smooth': z_quad_usmooth,
        # 'z_quad_noise': z_quad_noise,
        # 'z_quad_noise_gaussian-smooth': z_quad_noise_gsmooth,
        # 'z_quad_noise_uniform-smooth': z_quad_noise_usmooth,
        # 'z_cubic': z_cubic,
        # 'z_cubic_gaussian-smooth': z_cubic_gsmooth,
        # 'z_cubic_uniform-smooth': z_cubic_usmooth,
        # 'z_cubic_noise': z_cubic_noise,
        # 'z_cubic_noise_gaussian-smooth': z_cubic_noise_gsmooth,
        # 'z_cubic_noise_uniform-smooth': z_cubic_noise_usmooth,
        # 'z_pchip': z_pchip,
        # 'z_pchip_gaussian-smooth': z_pchip_gsmooth,
        # 'z_pchip_uniform-smooth': z_pchip_usmooth,
        # 'z_pchip_noise': z_pchip_noise,
        # 'z_pchip_noise_gaussian-smooth': z_pchip_noise_gsmooth,
        # 'z_pchip_noise_uniform-smooth': z_pchip_noise_usmooth,
        # 'z_aki': z_aki,
        # 'z_aki_gaussian-smooth': z_aki_gsmooth,
        # 'z_aki_uniform-smooth': z_aki_usmooth,
        # 'z_aki_noise': z_aki_noise,
        # 'z_aki_noise_gaussian-smooth': z_aki_noise_gsmooth,
        # 'z_aki_noise_uniform-smooth': z_aki_noise_usmooth
    }.items():
        print('----- %s -----' % k)
        images = Gs.run(v, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5,
                        out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)  # (1725, 1, 1024, 1024)
        images1 = np.swapaxes(images, axis1=0, axis2=2)  # [1024, 1, 1725, 1024]
        images2 = np.swapaxes(images, axis1=0, axis2=3)  # [1024, 1, 1024, 1725]

        # ths1 = []
        # ths2 = []
        # for img in images:
        #     pixel_set = np.expand_dims(img.flatten(), axis=-1)
        #     kmeans = KMeans(n_clusters=2, random_state=0, verbose=0).fit(pixel_set)
        #     cluster1 = np.squeeze(pixel_set[kmeans.labels_ == 0])
        #     # print('Class 1')
        #     # print(np.max(cluster1))
        #     # print(np.min(cluster1))
        #     cluster2 = np.squeeze(pixel_set[kmeans.labels_ == 1])
        #     # print('Class 2')
        #     # print(np.max(cluster2))
        #     # print(np.min(cluster2))
        #     # cluster3 = np.squeeze(pixel_set[kmeans.labels_ == 2])
        #     # print('Class 3')
        #     # print(np.max(cluster3))
        #     # print(np.min(cluster3))
        #     max = [np.max(cluster1), np.max(cluster2)]
        #     max.sort()
        #     ths1.append(max[0])
        #     ths2.append(max[1])
        # print(np.bincount(ths1, minlength=256))
        # print(np.bincount(ths2, minlength=256))
        # print(np.mean(ths1))
        # print(np.mean(ths2))
        # ths1 = stats.mode(ths1)[0][0]
        # ths2 = stats.mode(ths2)[0][0]
        # print('Thres1: ' + str(ths1))
        # print('Thres2: ' + str(ths2))
        ths1 = 95
        ths2 = 143

        seg = np.zeros_like(images, dtype=np.float32)
        seg[images <= ths1] = 0
        seg[np.logical_and(images >= ths1 + 1, images <= ths2)] = 127.5
        seg[images >= ths2 + 1] = 255

        seg1 = np.swapaxes(seg, axis1=0, axis2=2)  # [1024, 1, 1725, 1024]
        seg2 = np.swapaxes(seg, axis1=0, axis2=3)  # [1024, 1, 1024, 1725]

        # for i in range(int(np.ceil(final_images.shape[0] / minibatch_size))):
        misc.save_image_grid(images1[600:601],
                             os.path.join(result_subdir, 's1-%s%06d.png' % (k, 600)),
                             [0, 255], grid_size)
        misc.save_image_grid(images2[600:601],
                             os.path.join(result_subdir, 's2-%s%06d.png' % (k, 600)),
                             [0, 255], grid_size)
        misc.save_image_grid(seg1[600:601],
                             os.path.join(result_subdir, 's1-seg-%s%06d.png' % (k, 600)),
                             [0, 255], grid_size)
        misc.save_image_grid(seg2[600:601],
                             os.path.join(result_subdir, 's2-seg-%s%06d.png' % (k, 600)),
                             [0, 255], grid_size)
        misc.save_image_grid(images1[600:601, :, 1530:1720],
                             os.path.join(result_subdir, 's1-test-%s%06d.png' % (k, 600)),
                             [0, 255], grid_size)
        misc.save_image_grid(images2[600:601, :, :, 1530:1720],
                             os.path.join(result_subdir, 's2-test-%s%06d.png' % (k, 600)),
                             [0, 255], grid_size)

        loss = np.mean(np.square(real_imgs - (images.astype(np.float32) / 127.5 - 1)), axis=(1, 2, 3))
        ind = np.where(loss > 0.08)[0]
        print('mean loss ' + str(np.mean(loss)))
        PSNR = np.mean(10 * np.log10(1 / loss))
        print('mean PSNR ' + str(PSNR))
        print(
            'mean test loss ' + str(np.mean(loss[np.delete(np.arange(1530, 1720, 1), np.arange(0, 190, 9), axis=0)])))

        print('mean train loss ' + str(
            np.mean(loss[np.concatenate((np.arange(0, 1530, 1), np.arange(1530, 1720, 9)),
                                        axis=0)])))
        for i in range(len(ind)):
            print('slice ind %d loss %f' % (ind[i], loss[ind[i]]))

        loss = np.mean(np.square(seg_real - seg), axis=(1, 2, 3))
        print('mean loss for segmented image ' + str(np.mean(loss) / 255 / 255))
        PSNR = np.mean(10 * np.log10(255 ** 2 / loss))
        print('mean PSNR for segmented image ' + str(PSNR))

        # # one hot
        # temp1 = np.zeros((seg_real.shape[0], 1, seg_real.shape[2], seg_real.shape[3]), dtype=np.float32)
        # temp1[seg_real == 0] = 1
        # temp2 = np.zeros((seg_real.shape[0], 1, seg_real.shape[2], seg_real.shape[3]), dtype=np.float32)
        # temp2[seg_real == 127.5] = 1
        # temp3 = np.zeros((seg_real.shape[0], 1, seg_real.shape[2], seg_real.shape[3]), dtype=np.float32)
        # temp3[seg_real == 255] = 1
        # one_hot_real = np.concatenate((temp1, temp2, temp3), axis=1)
        # temp1 = np.zeros((seg_real.shape[0], 1, seg_real.shape[2], seg_real.shape[3]), dtype=np.float32)
        # temp1[seg == 0] = 1
        # temp2 = np.zeros((seg_real.shape[0], 1, seg_real.shape[2], seg_real.shape[3]), dtype=np.float32)
        # temp2[seg == 127.5] = 1
        # temp3 = np.zeros((seg_real.shape[0], 1, seg_real.shape[2], seg_real.shape[3]), dtype=np.float32)
        # temp3[seg == 255] = 1
        # one_hot = np.concatenate((temp1, temp2, temp3), axis=1)
        # ce = -np.mean(one_hot_real * np.log(one_hot + 0.00001))
        # print('mean CE for segmented image ' + str(ce))

        # for i in range(int(np.ceil(final_images.shape[0] / minibatch_size))):
        misc.save_image_grid(real_imgs[ind],
                             os.path.join(result_subdir, 'real-%s.png' % k),
                             [-1, 1])
        misc.save_image_grid(images[ind],
                             os.path.join(result_subdir, 'fake-%s.png' % k),
                             [0, 255])

        if k == 'z_linear':
            mark = np.ones_like(images) * 255
            mark[xp] = images[xp]

            mark1 = np.swapaxes(mark, axis1=0, axis2=2)  # [1024, 1, 1725, 1024]
            mark2 = np.swapaxes(mark, axis1=0, axis2=3)  # [1024, 1, 1725, 1024]
            misc.save_image_grid(mark1[600:601],
                                 os.path.join(result_subdir, 's1-%s%06d.png' % ('mark', 600)),
                                 [0, 255], grid_size)
            misc.save_image_grid(mark2[600:601],
                                 os.path.join(result_subdir, 's2-%s%06d.png' % ('mark', 600)),
                                 [0, 255], grid_size)
            Ind = np.sort(np.append(np.arange(432, 568, 9), np.arange(436, 568, 9), axis=0), axis=0)
            print(Ind)
            # anim_file = result_subdir + '/train-%s.gif' % k
            # import imageio
            #
            # with imageio.get_writer(anim_file, mode='I', duration=list(np.append(np.ones(len(Ind)) * 0.5, 3)),
            #                         subrectangles=True) as writer:
            cmap_reversed = matplotlib.cm.get_cmap('gray')
            for i in range(len(Ind)):
                print(np.max((images[Ind[i], 0].astype(np.float32) / 127.5 - 1) - real_imgs[Ind[i], 0]))
                print(np.min((images[Ind[i], 0].astype(np.float32) / 127.5 - 1) - real_imgs[Ind[i], 0]))

                plt.figure(figsize=(16, 6))
                plt.subplot(1, 3, 1)
                plt.imshow(images[Ind[i], 0], cmap='gray', vmin=0, vmax=255)
                plt.xticks([])
                plt.yticks([])
                plt.box(False)
                # if i % 2 == 0:
                #     plt.xlabel('Inverted slice')
                # if i % 2 == 1:
                #     plt.xlabel('Interpolated slice')
                plt.subplot(1, 3, 2)
                plt.imshow(real_imgs[Ind[i], 0], cmap='gray', vmin=-1, vmax=1)
                plt.xticks([])
                plt.yticks([])
                plt.box(False)
                # plt.xlabel('Ground truth')
                # plt.title('Slice %d' % (Ind[i]+1))
                plt.subplot(1, 3, 3)
                plt.imshow((images[Ind[i], 0].astype(np.float32) / 127.5 - 1) - real_imgs[Ind[i], 0],
                           cmap=cmap_reversed, vmin=-1, vmax=1)
                plt.xticks([])
                plt.yticks([])
                plt.box(False)
                # plt.xlabel('Prediction error')
                plt.tight_layout()
                plt.savefig(result_subdir + '/train-slice%d.png' % Ind[i])
                plt.close()
                # writer.append_data(imageio.imread(result_subdir + '/train-slice%d.pdf' % Ind[i]))
                # writer.append_data(imageio.imread(result_subdir + '/train-slice%d.pdf' % Ind[i]))

            plt.figure(figsize=(16, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(images[Ind[i], 0], cmap='gray', vmin=0, vmax=255)
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            # if i % 2 == 0:
            #     plt.xlabel('Inverted slice')
            # if i % 2 == 1:
            #     plt.xlabel('Interpolated slice')
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(real_imgs[Ind[i], 0], cmap='gray', vmin=-1, vmax=1)
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            # plt.xlabel('Ground truth')
            # plt.title('Slice %d' % (Ind[i] + 1))
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow((images[Ind[i], 0].astype(np.float32) / 127.5 - 1) - real_imgs[Ind[i], 0],
                       cmap=cmap_reversed, vmin=-1, vmax=1)
            plt.xticks([])
            plt.yticks([])
            # plt.xlabel('Prediction error')
            plt.colorbar()
            plt.box(False)
            plt.tight_layout()
            plt.savefig(result_subdir + '/train-slice%d-colorbar.png' % Ind[i])
            plt.close()

            Ind = np.sort(np.append(np.arange(1584, 1720, 9), np.arange(1588, 1720, 9), axis=0), axis=0)
            print(Ind)
            # anim_file = result_subdir + '/test-%s.gif' % k
            #
            # with imageio.get_writer(anim_file, mode='I', duration=list(np.append(np.ones(len(Ind)) * 0.5, 3)),
            #                         subrectangles=True) as writer:
            for i in range(len(Ind)):
                print(np.max((images[Ind[i], 0].astype(np.float32) / 127.5 - 1) - real_imgs[Ind[i], 0]))
                print(np.min((images[Ind[i], 0].astype(np.float32) / 127.5 - 1) - real_imgs[Ind[i], 0]))
                plt.figure(figsize=(16, 6))
                plt.subplot(1, 3, 1)
                plt.imshow(images[Ind[i], 0], cmap='gray', vmin=0, vmax=255)
                plt.xticks([])
                plt.yticks([])
                plt.box(False)
                # if i % 2 == 0:
                #     plt.xlabel('Inverted slice')
                # if i % 2 == 1:
                #     plt.xlabel('Interpolated slice')
                plt.subplot(1, 3, 2)
                plt.imshow(real_imgs[Ind[i], 0], cmap='gray', vmin=-1, vmax=1)
                plt.xticks([])
                plt.yticks([])
                plt.box(False)
                # plt.xlabel('Ground truth')
                # plt.title('Slice %d' % (Ind[i]+1))
                plt.subplot(1, 3, 3)
                plt.imshow((images[Ind[i], 0].astype(np.float32) / 127.5 - 1) - real_imgs[Ind[i], 0],
                           cmap=cmap_reversed, vmin=-1, vmax=1)
                plt.xticks([])
                plt.yticks([])
                plt.box(False)
                # plt.xlabel('Prediction error')
                plt.tight_layout()
                plt.savefig(result_subdir + '/test-slice%d.png' % Ind[i])
                plt.close()
            #     writer.append_data(imageio.imread(result_subdir + '/test-slice%d.pdf' % Ind[i]))
            # writer.append_data(imageio.imread(result_subdir + '/test-slice%d.pdf' % Ind[i]))

    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()


def interpolation_between_images(grid_size=[1, 1], window_size=8, start=3, end=1720):
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)

    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)
    real_imgs, _ = training_set.get_minibatch_np(minibatch_size=end)

    xp = np.arange(start, end, window_size)

    # linear interp
    f = interpolate.interp1d(xp, real_imgs[xp], kind='linear', axis=0,
                             fill_value='extrapolate')
    full_img_linear = f(np.arange(0, end, 1))

    for k, v in {
        # 'nn': full_img_nn,
        'linear': full_img_linear,
        # 'cubic': full_img_cubic,
    }.items():
        print('----- %s -----' % k)

        loss = np.mean(np.square((real_imgs / 127.5 - 1) - (v.astype(np.float32) / 127.5 - 1)), axis=(1, 2, 3))
        PSNR = np.mean(10 * np.log10(1 / loss))
        print('mean PSNR ' + str(PSNR))
        ind = np.where(loss > 0.08)[0]
        for i in range(len(ind)):
            print('slice ind %d loss %f' % (ind[i], loss[ind[i]]))
        print(
            'mean loss ' + str(np.mean(loss)))
        print(
            'mean interp loss ' + str(np.mean(loss[np.delete(np.arange(0, end, 1), xp, axis=0)])))

        ths1 = []
        ths2 = []
        for img in v.astype(np.int32):
            pixel_set = np.expand_dims(img.flatten(), axis=-1)
            kmeans = KMeans(n_clusters=3, random_state=0, verbose=0).fit(pixel_set)
            cluster1 = np.squeeze(pixel_set[kmeans.labels_ == 0])
            # print('Class 1')
            # print(np.max(cluster1))
            # print(np.min(cluster1))
            cluster2 = np.squeeze(pixel_set[kmeans.labels_ == 1])
            # print('Class 2')
            # print(np.max(cluster2))
            # print(np.min(cluster2))
            cluster3 = np.squeeze(pixel_set[kmeans.labels_ == 2])
            # print('Class 3')
            # print(np.max(cluster3))
            # print(np.min(cluster3))
            max = [np.max(cluster1), np.max(cluster2), np.max(cluster3)]
            max.sort()
            ths1.append(max[0])
            ths2.append(max[1])
        print(np.bincount(ths1, minlength=256))
        print(np.bincount(ths2, minlength=256))
        print(np.mean(ths1))
        print(np.mean(ths2))
        ths1 = stats.mode(ths1)[0][0]
        ths2 = stats.mode(ths2)[0][0]
        print('Thres1: ' + str(ths1))
        print('Thres2: ' + str(ths2))

        # for i, img in enumerate(v):
        #     misc.save_image_grid(img,
        #                          os.path.join(result_subdir, 'fakes%04d.jpeg' % i),
        #                          [0, 255], grid_size)

    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()


def generate_thirdD_real_images(grid_size=[1, 1], minibatch_size=8):
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)

    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)
    real_imgs, _ = training_set.get_minibatch_np(minibatch_size=1725)
    print(real_imgs.shape)
    real_imgs1 = np.swapaxes(real_imgs, axis1=0, axis2=2)
    print(real_imgs1.shape)
    # for i in range(int(np.ceil(real_imgs.shape[0] / minibatch_size))):
    misc.save_image_grid(real_imgs1[600:601],
                         os.path.join(result_subdir, 's1-%s%06d.png' % ('real', 600)),
                         [0, 255], grid_size)

    real_imgs2 = np.swapaxes(real_imgs, axis1=0, axis2=3)
    print(real_imgs2.shape)
    # for i in range(int(np.ceil(real_imgs.shape[0] / minibatch_size))):
    misc.save_image_grid(real_imgs2[600:601],
                         os.path.join(result_subdir, 's2-%s%06d.png' % ('real', 600)),
                         [0, 255], grid_size)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()


def generate_fake_labelled_images(run_id, label, snapshot=None, grid_size=[1, 1],
                                  num_pngs=1, image_shrink=1, png_prefix=None,
                                  random_seed=1000, minibatch_size=8):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    if png_prefix is None:
        png_prefix = misc.get_id_string_for_network_pkl(network_pkl) + '-'
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    for png_idx in range(num_pngs):
        print('Generating png %d / %d...' % (png_idx, num_pngs))
        latents = misc.random_latents(np.prod(grid_size), Gs, random_state=random_state)
        labels = np.repeat(np.expand_dims(label, axis=0), latents.shape[0], axis=0).astype(np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5,
                        out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        misc.save_image_grid(images, os.path.join(result_subdir, '%s%06d.png' % (png_prefix, png_idx)), [0, 255],
                             grid_size)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()


# ----------------------------------------------------------------------------
# Generate MP4 video of random interpolations using a previously trained network.
# To run, uncomment the appropriate line in config.py and launch train.py.

def generate_interpolation_video(run_id, snapshot=None, grid_size=[1, 1], image_shrink=1, image_zoom=1,
                                 duration_sec=30.0, transit_sec=5, smoothing_sec=1.0, mp4=None, mp4_fps=10,
                                 mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8):
    assert duration_sec % transit_sec == 0, "duration_sec should be divisible by transit_sec"
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    if mp4 is None:
        mp4 = misc.get_id_string_for_network_pkl(network_pkl) + '-lerp.mp4'
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    print('Generating latent vectors...')
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:]  # [frame, image, channel, component]
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents,
                                                [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape),
                                                mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # all_latents = np.zeros([0, np.prod(grid_size)] + Gs.input_shape[1:])
    # num_transition = int(duration_sec / transit_sec)
    # shape = [num_transition+1, np.prod(grid_size)] + Gs.input_shape[1:]  # [2, image, channel, component]
    # end_latents = random_state.randn(*shape).astype(np.float32)
    # for i in range(num_transition):
    #     z0 = np.expand_dims(end_latents[i], axis=0)  # [1, image, channel, component]
    #     zt = np.expand_dims(end_latents[i+1], axis=0)  # [1, image, channel, component]
    #     for s in range(transit_sec * mp4_fps):
    #         zs = (zt - z0) / (transit_sec * mp4_fps - 0) * s + z0  # [1, image, channel, component]
    #         all_latents = np.append(all_latents, zs, axis=0)
    # all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        labels = np.zeros([latents.shape[0], 0], np.float32)
        images = Gs.run(latents, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5,
                        out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8)
        grid = misc.create_image_grid(images, grid_size).transpose(1, 2, 0)  # HWC
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)  # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor  # pip install moviepy
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, mp4),
                                                                                fps=mp4_fps, codec='libx264',
                                                                                bitrate=mp4_bitrate)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()


def generate_interpolation_image(run_id, snapshot=None, image_shrink=1, random_seed=1000, minibatch_size=16,
                                 start=0, window_size=9, end=1720):
    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    random_state = np.random.RandomState(random_seed)

    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)

    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    print('Generating latent vectors...')
    shape = [len(np.arange(start, end, window_size))] + Gs.input_shape[1:]  # [frame, component]
    z = random_state.randn(*shape).astype(np.float32)

    labels = np.zeros([z.shape[0], 0], np.float32)
    image = np.squeeze(
        Gs.run(z, labels, minibatch_size=minibatch_size, num_gpus=config.num_gpus, out_mul=127.5,
               out_add=127.5, out_shrink=image_shrink, out_dtype=np.uint8))
    # linear interp
    f = interpolate.interp1d(np.arange(start, end, window_size), image, kind='linear', axis=0,
                             fill_value='extrapolate')
    image_linear = f(np.arange(0, end, 1))

    ind_array = np.linspace(start=0, stop=image_linear.shape[1] - 1, num=5, dtype=np.int)
    for ind in ind_array:
        misc.save_image_grid(np.expand_dims(image_linear[:, ind, :], axis=0),
                             os.path.join(result_subdir, '%s%04d.png' % ('axis1-linear-', ind)),
                             [0, 255],
                             grid_size=(1, 1))
    ind_array = np.linspace(start=0, stop=image_linear.shape[2] - 1, num=5, dtype=np.int)
    for ind in ind_array:
        misc.save_image_grid(np.expand_dims(image_linear[:, :, ind], axis=0),
                             os.path.join(result_subdir, '%s%04d.png' % ('axis2-linear-', ind)),
                             [0, 255],
                             grid_size=(1, 1))

    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()


# ----------------------------------------------------------------------------
# Generate MP4 video of training progress for a previous training run.
# To run, uncomment the appropriate line in config.py and launch train.py.

def generate_training_video(run_id, duration_sec=20.0, time_warp=1.5, mp4=None, mp4_fps=30, mp4_codec='libx265',
                            mp4_bitrate='16M'):
    src_result_subdir = misc.locate_result_subdir(run_id)
    if mp4 is None:
        mp4 = os.path.basename(src_result_subdir) + '-train.mp4'

    # Parse log.
    times = []
    snaps = []  # [(png, kimg, lod), ...]
    with open(os.path.join(src_result_subdir, 'log.txt'), 'rt') as log:
        for line in log:
            k = re.search(r'kimg ([\d\.]+) ', line)
            l = re.search(r'lod ([\d\.]+) ', line)
            t = re.search(r'time (\d+d)? *(\d+h)? *(\d+m)? *(\d+s)? ', line)
            if k and l and t:
                k = float(k.group(1))
                l = float(l.group(1))
                t = [int(t.group(i)[:-1]) if t.group(i) else 0 for i in range(1, 5)]
                t = t[0] * 24 * 60 * 60 + t[1] * 60 * 60 + t[2] * 60 + t[3]
                png = os.path.join(src_result_subdir, 'fakes%06d.png' % int(np.floor(k)))
                if os.path.isfile(png):
                    times.append(t)
                    snaps.append((png, k, l))
    assert len(times)

    # Frame generation func for moviepy.
    png_cache = [None, None]  # [png, img]

    def make_frame(t):
        wallclock = ((t / duration_sec) ** time_warp) * times[-1]
        png, kimg, lod = snaps[max(bisect.bisect(times, wallclock) - 1, 0)]
        if png_cache[0] == png:
            img = png_cache[1]
        else:
            img = cv2.imread(png)
            while img.shape[1] > 1920 or img.shape[0] > 1080:
                img = img.astype(np.float32).reshape(img.shape[0] // 2, 2, img.shape[1] // 2, 2, -1).mean(axis=(1, 3))
            png_cache[:] = [png, img]
        img = misc.draw_text_label(img, 'lod %.2f' % lod, 16, img.shape[0] - 4, alignx=0.0, aligny=1.0)
        img = misc.draw_text_label(img, misc.format_time(int(np.rint(wallclock))), img.shape[1] // 2, img.shape[0] - 4,
                                   alignx=0.5, aligny=1.0)
        img = misc.draw_text_label(img, '%.0f kimg' % kimg, img.shape[1] - 16, img.shape[0] - 4, alignx=1.0, aligny=1.0)
        return img

    # Generate video.
    import moviepy.editor  # pip install moviepy
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    moviepy.editor.VideoClip(make_frame, duration=duration_sec).write_videofile(os.path.join(result_subdir, mp4),
                                                                                fps=mp4_fps, codec='libx264',
                                                                                bitrate=mp4_bitrate)
    open(os.path.join(result_subdir, '_done.txt'), 'wt').close()


# ----------------------------------------------------------------------------
# Evaluate one or more metrics for a previous training run.
# To run, uncomment one of the appropriate lines in config.py and launch train.py.

def evaluate_metrics(run_id, log, metrics, num_images, real_passes, minibatch_size=None):
    metric_class_names = {
        'swd': 'metrics.sliced_wasserstein.API',
        'fid': 'metrics.frechet_inception_distance.API',
        'is': 'metrics.inception_score.API',
        'msssim': 'metrics.ms_ssim.API',
    }

    # Locate training run and initialize logging.
    result_subdir = misc.locate_result_subdir(run_id)
    snapshot_pkls = misc.list_network_pkls(result_subdir, include_final=False)
    assert len(snapshot_pkls) >= 1
    log_file = os.path.join(result_subdir, log)
    print('Logging output to', log_file)
    misc.set_output_log_file(log_file)

    # Initialize dataset and select minibatch size.
    dataset_obj, mirror_augment = misc.load_dataset_for_previous_run(result_subdir, verbose=True, shuffle_mb=0)
    if minibatch_size is None:
        minibatch_size = np.clip(8192 // dataset_obj.shape[1], 4, 256)

    # Initialize metrics.
    metric_objs = []
    for name in metrics:
        class_name = metric_class_names.get(name, name)
        print('Initializing %s...' % class_name)
        class_def = tfutil.import_obj(class_name)
        image_shape = [3] + dataset_obj.shape[1:]
        obj = class_def(num_images=num_images, image_shape=image_shape, image_dtype=np.uint8,
                        minibatch_size=minibatch_size)
        tfutil.init_uninited_vars()
        mode = 'warmup'
        obj.begin(mode)
        for idx in range(10):
            obj.feed(mode, np.random.randint(0, 256, size=[minibatch_size] + image_shape, dtype=np.uint8))
        obj.end(mode)
        metric_objs.append(obj)

    # Print table header.
    print()
    print('%-10s%-12s' % ('Snapshot', 'Time_eval'), end='')
    for obj in metric_objs:
        for name, fmt in zip(obj.get_metric_names(), obj.get_metric_formatting()):
            print('%-*s' % (len(fmt % 0), name), end='')
    print()
    print('%-10s%-12s' % ('---', '---'), end='')
    for obj in metric_objs:
        for fmt in obj.get_metric_formatting():
            print('%-*s' % (len(fmt % 0), '---'), end='')
    print()

    # Feed in reals.
    for title, mode in [('Reals', 'reals'), ('Reals2', 'fakes')][:real_passes]:
        print('%-10s' % title, end='')
        time_begin = time.time()
        labels = np.zeros([num_images, dataset_obj.label_size], dtype=np.float32)
        [obj.begin(mode) for obj in metric_objs]
        for begin in range(0, num_images, minibatch_size):
            end = min(begin + minibatch_size, num_images)
            images, labels[begin:end] = dataset_obj.get_minibatch_np(end - begin)
            if mirror_augment:
                images = misc.apply_mirror_augment(images)
            if images.shape[1] == 1:
                images = np.tile(images, [1, 3, 1, 1])  # grayscale => RGB
            [obj.feed(mode, images) for obj in metric_objs]
        results = [obj.end(mode) for obj in metric_objs]
        print('%-12s' % misc.format_time(time.time() - time_begin), end='')
        for obj, vals in zip(metric_objs, results):
            for val, fmt in zip(vals, obj.get_metric_formatting()):
                print(fmt % val, end='')
        print()

    # Evaluate each network snapshot.
    for snapshot_idx, snapshot_pkl in enumerate(reversed(snapshot_pkls)):
        prefix = 'network-snapshot-'
        postfix = '.pkl'
        snapshot_name = os.path.basename(snapshot_pkl)
        assert snapshot_name.startswith(prefix) and snapshot_name.endswith(postfix)
        snapshot_kimg = int(snapshot_name[len(prefix): -len(postfix)])

        print('%-10d' % snapshot_kimg, end='')
        mode = 'fakes'
        [obj.begin(mode) for obj in metric_objs]
        time_begin = time.time()
        with tf.Graph().as_default(), tfutil.create_session(config.tf_config).as_default():
            G, D, Gs = misc.load_pkl(snapshot_pkl)
            for begin in range(0, num_images, minibatch_size):
                end = min(begin + minibatch_size, num_images)
                latents = misc.random_latents(end - begin, Gs)
                images = Gs.run(latents, labels[begin:end], num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5,
                                out_dtype=np.uint8)
                if images.shape[1] == 1:
                    images = np.tile(images, [1, 3, 1, 1])  # grayscale => RGB
                [obj.feed(mode, images) for obj in metric_objs]
        results = [obj.end(mode) for obj in metric_objs]
        print('%-12s' % misc.format_time(time.time() - time_begin), end='')
        for obj, vals in zip(metric_objs, results):
            for val, fmt in zip(vals, obj.get_metric_formatting()):
                print(fmt % val, end='')
        print()
    print()


# ----------------------------------------------------------------------------

# invert GAN
def invert_gan(
        run_id,
        resume_invert_id=None,
        initial_learning_rate=0.001,
        lowest_learning_rate=0.00001,
        reals_np=None,
        labels_np=None,
        num_sampling=1000,
        num_images=8,
        start_img_id=0,
        minibatch=8,
        snapshot=None,
        total_step=100,
        mini_step=100,
        decay_rate=0.95,
        decay_steps=10.0,
        drange_net=[-1, 1],
        # Dynamic range used when feeding image data to the networks.
        image_snapshot_steps=10,
        # How often to export image snapshots?
        z_snapshot_steps=100,
        # How often to export z snapshots?
        print_progress_steps=100,
        random_seed=1000
):
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    # assert start_img_id % minibatch == 0, 'start_img-id shall be divisible by minibatch'

    with tf.device('/gpu:0'):
        network_pkl = misc.locate_network_pkl(run_id, snapshot)
        print('Loading network from "%s"...' % network_pkl)
        G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    print('Setting up result dir...')
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)

    # if reals_np is None:
    #     training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)
    #     if start_img_id != 0:
    #         for _ in range(int(start_img_id / minibatch)):
    #             _, _ = training_set.get_minibatch_np(minibatch_size=minibatch)
    # if reals_np is None:
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)
    reals, labels = training_set.get_minibatch_np(minibatch_size=1720)
    # else:
    #     reals = reals_np[(start_img_id + index * minibatch):(start_img_id + (index + 1) * minibatch)]
    #     labels = labels_np[(start_img_id + index * minibatch):(start_img_id + (index + 1) * minibatch)]

    # reals = reals[np.concatenate((np.arange(0, 512, 9), np.arange(512, 701, 9), np.arange(701, 1725, 9)))]
    reals = reals[np.arange(0, 1720, 9)]
    # reals = reals[
    #     [1034, 1035, 1037, 1039, 1040, 1043, 1044, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1057, 1061,
    #      1063, 1079]]
    # labels = labels[
    #     [1034, 1035, 1037, 1039, 1040, 1043, 1044, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1057, 1061,
    #      1063, 1079]]
    # normalize reals
    reals = np.array(reals / 127.5 - 1, dtype=np.float32)

    # labels = labels[np.concatenate((np.arange(0, 512, 9), np.arange(512, 701, 9), np.arange(701, 1725, 9)))]
    labels = labels[np.arange(0, 1720, 9)]

    num_images = reals.shape[0]
    print('number of images:' + str(reals.shape[0]))

    num_trials = np.ceil(num_images / minibatch).astype(np.int32)
    loss_history = np.zeros((num_trials, total_step), dtype=np.float32)
    random_state = np.random.RandomState(random_seed)
    for index in range(num_trials):
        print('...... trial %d/%d ......' % (index + 1, num_trials))
        reals_batch = reals[index * minibatch:(index + 1) * minibatch]
        labels_batch = labels[index * minibatch:(index + 1) * minibatch]
        # reals_batch, labels_batch = training_set.get_minibatch_np(minibatch_size=minibatch)
        # reals_batch = reals_batch / 127.5 - 1
        min_loss = np.ones(reals_batch.shape[0]) * np.inf
        z_final = np.zeros((reals_batch.shape[0], 512), dtype=np.float32)

        if resume_invert_id is None:
            # random sampling
            print('Start sampling method')
            start = time.time()
            z_samples = random_state.randn(num_sampling, reals_batch.shape[0], *Gs.input_shape[1:]).astype(np.float32)
            # z_samples = np.repeat(z_samples, repeats=reals.shape[0], axis=1)
            error_log = np.zeros((z_samples.shape[0], z_samples.shape[1]), dtype=np.float32)

            for ind, z in enumerate(z_samples):
                fakes = Gs.run(z, labels_batch, minibatch_size=minibatch)
                error_log[ind, :] = np.mean(np.square(reals_batch - fakes), axis=(1, 2, 3))
            sort_ind = np.argsort(error_log, axis=0)
            sorted_error_log = np.take_along_axis(error_log, sort_ind, axis=0)
            sorted_z_samples = np.take_along_axis(z_samples, np.expand_dims(sort_ind, axis=[2]), axis=0)

            print('lowest initial error')
            print(sorted_error_log[0])
            print('Finish sampling method in %.1f sec' % (time.time() - start))

            # plt.figure()
            # plt.hist(error_log, density=True, bins=np.arange(0.05, 0.20, 0.005))
            # plt.xlabel('Loss')
            # plt.ylabel('Density')
            # plt.savefig(os.path.join(result_subdir, 'error_hist_image%d.png' % ind))
            # plt.tight_layout()
            # plt.close()

            # z_optimals = np.repeat(sorted_z_samples[10 * index][(int((num_images + 1) / 2) - 1):int((num_images + 1) / 2)],
            #                        repeats=num_images,
            #                        axis=0)
            # z_optimals = np.ones((reals.shape[0], *Gs.input_shape[1:]), dtype=np.float32)
            z_optimals = sorted_z_samples[0]
        else:
            dir_ = misc.locate_result_subdir(resume_invert_id)
            z_optimals = loadmat(os.path.join(dir_, 'z_final_trial%d.mat' % index))['z']
            # z_optimals = np.repeat(z_optimals, repeats=reals.shape[0], axis=0)

        misc.save_image_grid(reals_batch, os.path.join(result_subdir, 'reals_trial%d.png' % index), drange=drange_net)

        print('Building TensorFlow graph...')
        with tf.name_scope('Inputs'):
            lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])
            latent_in = tf.Variable(initial_value=z_optimals,
                                    trainable=True,
                                    shape=z_optimals.shape,
                                    name='latent_in',
                                    dtype=tf.float32)
        z_opt = tf.train.AdamOptimizer(learning_rate=lrate_in)
        with tf.name_scope('GPU0'), tf.device('/gpu:0'):
            with tf.name_scope('z_loss'):
                cal_loss_op = tfutil.call_func_by_name(Gs=Gs,
                                                       D=D,
                                                       reals=reals_batch,
                                                       labels=labels_batch,
                                                       latents=latent_in,
                                                       **config.z_loss)
                LOSS = tf.reduce_mean(cal_loss_op)
            # with tf.name_scope('z_loss'):
            #     fake_images_out = Gs_gpu.get_output_for(latent_in, labels, is_training=False)
            #     with tf.name_scope('MSE'):
            #         mse = tf.reduce_mean(tf.square(fake_images_out - reals))
            grad_and_var = z_opt.compute_gradients(LOSS, [latent_in])
            z_op = z_opt.apply_gradients(grad_and_var)
            # latent_in = (latent_in - tf.math.reduce_mean(latent_in)) / (tf.math.reduce_std(latent_in) + 1e-8)

        # summary_log = tf.summary.FileWriter(result_subdir)
        savemat(os.path.join(result_subdir, 'z_sampling_trial%d.mat' % index), {'z_sampling': z_optimals})
        fake_images = Gs.run(z_optimals, labels_batch, minibatch_size=minibatch)
        misc.save_image_grid(fake_images, os.path.join(result_subdir, 'fake_z_sampling_trial%d.png' % index),
                             drange=drange_net)

        print('Training...')
        tfutil.run(latent_in.initializer)
        tfutil.run(tf.variables_initializer(z_opt.variables()))

        # savemat(os.path.join(result_subdir, 'z%06d_trial%d.mat' % (0, index)), {'z': tfutil.run(latent_in)})
        # fake_images = Gs.run(tfutil.run(latent_in), labels, minibatch_size=minibatch)
        # misc.save_image_grid(fake_images, os.path.join(result_subdir, 'fakes%06d_trial%d.png' % (0, index)),
        #                      drange=drange_net)

        # count = 0
        for cur_step in range(total_step):
            z_temp = tfutil.run(latent_in)
            decayed_learning_rate = initial_learning_rate * decay_rate ** (cur_step // decay_steps)
            z_loss_vec, z_loss, _, _ = tfutil.run([cal_loss_op, LOSS, grad_and_var, z_op],
                                                  {lrate_in: max(decayed_learning_rate, lowest_learning_rate)})
            loss_history[index, cur_step] = z_loss

            z_final[np.where(z_loss_vec < min_loss)[0]] = np.copy(z_temp[np.where(z_loss_vec < min_loss)[0]])
            min_loss = np.minimum(min_loss, z_loss_vec)

            # print('Step %-8d loss: %-8.5f' % (cur_step + 1, z_loss))
            # if cur_step > 0 and loss_history[index, cur_step - 1] - loss_history[index, cur_step] < 0.00001:
            #     count = count + 1
            # else:
            #     count = 0
            # tfutil.autosummary('Progress/learning_rate', decayed_learning_rate)
            cur_step += 1
            # tfutil.save_summaries(summary_log, cur_step)
            done = (cur_step == total_step)
            if cur_step % print_progress_steps == 0 or done:
                print('step %-8d learning_rate %-8e loss %-8.5f' % (cur_step, decayed_learning_rate, z_loss))
            # # Save snapshots.
            # if cur_step % image_snapshot_steps == 0:
            #     fake_images = Gs.run(tfutil.run(latent_in), labels_batch, minibatch_size=minibatch)
            #     misc.save_image_grid(fake_images,
            #                          os.path.join(result_subdir, 'fakes%06d_trial%d.png' % (cur_step, index)),
            #                          drange=drange_net)
            # if cur_step % z_snapshot_steps == 0:
            #     savemat(os.path.join(result_subdir, 'z%06d_trial%d.mat' % (cur_step, index))
            #             , {'z': tfutil.run(latent_in)})
            # if count > 5 and cur_step > mini_step:
            #     break

        # save final results.
        z_temp = tfutil.run(latent_in)
        z_loss_vec = tfutil.run([cal_loss_op])
        z_final[np.where(z_loss_vec < min_loss)[0]] = np.copy(z_temp[np.where(z_loss_vec < min_loss)[0]])
        min_loss = np.minimum(min_loss, z_loss_vec)
        fake_images = Gs.run(z_final, labels_batch, minibatch_size=minibatch)
        misc.save_image_grid(fake_images,
                             os.path.join(result_subdir, 'fakes_final_trial%d.png' % index),
                             drange=drange_net)
        savemat(os.path.join(result_subdir, 'z_final_trial%d.mat' % index), {'z': z_final, 'min_loss': min_loss})
        # summary_log.close()

    plt.figure()
    for tt in range(num_trials):
        plt.plot(loss_history[tt], label='trial%d' % tt)
    plt.xlabel('Step #')
    plt.ylabel('MSE')
    # plt.yscale('log')
    plt.legend(loc='best')
    # plt.tight_layout()
    plt.savefig(os.path.join(result_subdir, 'history.png'))
    plt.close()
    savemat(os.path.join(result_subdir, 'loss.mat'), {'loss': loss_history})
    open(os.path.join(result_subdir, '_training-done.txt'), 'wt').close()


# ----------------------------------------------------------------------------

#  simulated annealing GAN
def SA_gan(
        run_id,
        reals=None,
        labels=None,
        num_sampling=1000,
        num_images=8,
        num_trials=20,
        minibatch=8,
        snapshot=None,
        P1=0.9,
        P2=0.1,
        Iter=100,
        Mo=512,
        Maxtime=51200,
        beta=1,
        drange_net=[-1, 1],
        random_seed=1000,
        T_func='cos'
):
    import random
    import matplotlib.pyplot as plt

    network_pkl = misc.locate_network_pkl(run_id, snapshot)
    print('Loading network from "%s"...' % network_pkl)
    G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    print('Setting up result dir...')
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)

    if reals is None:
        training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)
        reals, labels = training_set.get_minibatch_np(minibatch_size=num_images)
    # normalize reals
    reals = np.array(reals / 127.5 - 1, dtype=np.float32)

    # random sampling
    print('Start sampling method')
    start = time.time()
    random_state = np.random.RandomState(random_seed)
    z_samples = random_state.randn(num_sampling, reals.shape[0], *Gs.input_shape[1:]).astype(np.float32)
    error_log = np.zeros((z_samples.shape[0], reals.shape[0]), dtype=np.float32)

    for ind, z in enumerate(z_samples):
        fakes = Gs.run(z, labels, minibatch_size=minibatch)
        error_log[ind] = np.mean(np.square(reals - fakes), axis=(1, 2, 3))
    sort_ind = np.argsort(error_log, axis=0)
    sorted_error_log = np.take_along_axis(error_log, sort_ind, axis=0)
    sorted_z_samples = np.take_along_axis(z_samples, np.expand_dims(sort_ind, axis=-1), axis=0)

    print('lowest %d initial error' % num_trials)
    print(sorted_error_log[0:num_trials])
    print('Finish sampling method in %.1f sec' % (time.time() - start))

    To = np.zeros(reals.shape[0], dtype=np.float32)
    T2 = np.zeros(reals.shape[0], dtype=np.float32)
    alpha_list = np.zeros(reals.shape[0], dtype=np.float32)

    avgDeltaCost = np.sum(sorted_error_log - sorted_error_log[0:1], axis=0) / (sorted_error_log.shape[0] - 1)
    print('avgDeltaCost: %.4f' % avgDeltaCost)
    To = -avgDeltaCost / np.log(P1)
    T2 = -avgDeltaCost / np.log(P2)
    alpha_list = (np.log(P1) / np.log(P2)) ** (1 / Iter)
    for ind in range(reals.shape[0]):
        plt.figure()
        plt.hist(np.flatten(error_log[:, ind]), density=True, bins=np.arange(0.10, 0.20, 0.001))
        plt.xlabel('Loss')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(os.path.join(result_subdir, 'error_hist_image%d.png' % ind))
        plt.close()

    print('lowest %d initial error' % num_trials)
    print(sorted_error_log[0:num_trials])
    print('To')
    print(To)
    print('T2')
    print(T2)
    print('alpha')
    print(alpha_list)
    print('Finish sampling method in %.1f sec' % (time.time() - start))

    savemat(os.path.join(result_subdir, 'z_sampling.mat'), {'z': sorted_z_samples[0]})
    misc.save_image_grid(reals, os.path.join(result_subdir, 'reals.png'), drange=drange_net)
    fake_images = Gs.run(sorted_z_samples[0], labels, minibatch_size=minibatch)
    misc.save_image_grid(fake_images, os.path.join(result_subdir, 'fake_z_optimals.png'), drange=drange_net)

    def neighbor(S, scale, perturb_dim):
        # perturb_dim = np.random.randint(0, S.shape[1])
        S[:, perturb_dim] = S[:, perturb_dim] + np.random.randn(1) * scale
        return S

    # simulated annealing
    def SA(index, z_samples_):
        print('...... Image %d ......' % index)
        bestbestcost = np.inf
        real_ = reals[index:(index + 1)]
        label_ = labels[index:(index + 1)]
        Ti = To[index]
        Tf = T2[index]
        alpha = alpha_list[index]
        M = Mo
        record = np.zeros((z_samples_.shape[0], Maxtime + 1, 5), dtype=np.float32)
        for temp, CurS in enumerate(z_samples_):
            print('... z sample %d bestbest Cost %.4f...' % (temp, bestbestcost))
            scale = 1
            T = Ti
            CurS = np.expand_dims(CurS, axis=0)
            BestS = np.copy(CurS)
            fake_ = Gs.run(CurS, label_, minibatch_size=1)
            # fake_score_, _ = D.run(fake_, minibatch_size=1)
            # print(fake_score_)
            # real_score_, _ = D.run(real_, minibatch_size=1)
            # print(real_score_)
            CurCost = np.mean(np.square(real_ - fake_))
            # + np.abs(fake_score_ - real_score_) * 0.01
            BestCost = CurCost
            Time = 0
            i_ = 0
            t_ = 0
            dim = np.arange(z_samples_.shape[1])
            np.random.shuffle(dim)
            record[temp, t_] = [t_, CurCost, BestCost, P1, T]
            t_ = t_ + 1
            while Time < Maxtime:
                # if T >= Tf:
                #     flag = 1
                # else:
                #     if flag == 1:
                #         start_time = Time
                #         flag = 0
                #         print('start scale decay at time %d' % start_time)
                #     scale = ((Maxtime - Time) // M) / ((Maxtime - start_time) // M)
                # scale = 1-(Time // M) / (Maxtime // M)
                print('Time %d BestCost %.4f scale %.2f' % (Time, BestCost, scale))
                m = M
                while m > 0:
                    NewS = neighbor(CurS, scale, dim[i_])
                    fake_ = Gs.run(NewS, label_, minibatch_size=1)
                    # fake_score_, _ = D.run(fake_, minibatch_size=1)
                    # real_score_, _ = D.run(real_, minibatch_size=1)
                    NewCost = np.mean(np.square(fake_ - real_))  # + np.abs(fake_score_ - real_score_)
                    deltaCost = NewCost - CurCost
                    if deltaCost < 0:
                        CurS = np.copy(NewS)
                        CurCost = np.copy(NewCost)
                        if NewCost < BestCost:
                            BestCost = np.copy(NewCost)
                            BestS = np.copy(NewS)
                    else:
                        i_ = (i_ + 1) % z_samples_.shape[1]
                        if random.uniform(0, 1) < np.exp(-deltaCost / T):
                            CurS = np.copy(NewS)
                            CurCost = NewCost
                    m = m - 1
                    record[temp, t_] = [t_, CurCost, BestCost, np.exp(-deltaCost / T), T]
                    t_ = t_ + 1
                Time = Time + M
                if T_func == 'exp':
                    T = alpha * T
                if T_func == 'cos':
                    T = 0.5 * (Ti - Tf) * np.cos(np.pi * Time / Maxtime) + 0.5 * (Ti + Tf)
                M = beta * M

            if BestCost < bestbestcost:
                bestbestcost = BestCost
                bestbestS = np.copy(BestS)
                best_ind = temp

        plt.figure()
        for p in range(z_samples_.shape[0]):
            # plt.plot(record[p, :, 0], record[p, :, 1], 'b', label='CurCost')
            plt.plot(record[p, :, 0], record[p, :, 2], 'b')
        plt.plot(record[best_ind, :, 0], record[best_ind, :, 2], 'r')
        plt.xlabel('Iteration #')
        plt.ylabel('Best cost')
        plt.tight_layout()
        plt.savefig(os.path.join(result_subdir, 'all_trials_image%d.png' % index))
        plt.close()

        plt.figure()
        plt.subplot(121)
        plt.plot(record[best_ind, :, 0], record[best_ind, :, 1], '.', label='CurCost')
        plt.plot(record[best_ind, :, 0], record[best_ind, :, 2], label='BestCost')
        plt.legend(loc='best')
        plt.xlabel('Iteration #')
        plt.ylabel('Best/Cur Cost')
        plt.subplot(122)
        plt.plot(record[best_ind, :, 0], record[best_ind, :, 3], '.', label='probability')
        plt.plot(record[best_ind, :, 0], record[best_ind, :, 4], label='T')
        plt.legend(loc='best')
        plt.ylim([0, 1])
        plt.xlabel('Iteration #')
        plt.tight_layout()
        plt.savefig(os.path.join(result_subdir, 'history_image%d_best.png' % index))
        plt.close()

        return bestbestS

    print('\nStart optimization')
    t = time.time()
    z_SA = np.zeros_like(z_samples[0], dtype=np.float32)
    for ind in range(reals.shape[0]):
        # input2 = np.repeat(z_optimals[ind:(ind + 1)], repeats=num_trials, axis=0)
        # input2 = np.append(z_trial, z_optimals[ind:(ind + 1)], axis=0)
        input2 = sorted_z_samples[0:num_trials, ind]
        z_SA[ind:(ind + 1)] = SA(ind, input2)
    print('Finish optimization in {:f} seconds.\n'.format(time.time() - t))

    fake_images = Gs.run(z_SA, labels, minibatch_size=minibatch)
    misc.save_image_grid(fake_images, os.path.join(result_subdir, 'fake_z_SA.png'), drange=drange_net)
    savemat(os.path.join(result_subdir, 'z_SA.mat'), {'z': z_SA})


# ----------------------------------------------------------------------------

# invert GAN
def uni_invert_gan(
        run_id,
        initial_learning_rate=0.001,
        lowest_learning_rate=0.00001,
        reals_np=None,
        labels_np=None,
        start_img_id=0,
        uni_batch=512,
        num_images=None,
        num_sampling=1000,
        minibatch=16,
        snapshot=None,
        total_step=10000,
        mini_step=10000,
        decay_rate=0.9,
        decay_steps=1000.0,
        drange_net=[-1, 1],
        # Dynamic range used when feeding image data to the networks.
        image_snapshot_steps=10,
        # How often to export image snapshots?
        z_snapshot_steps=100,
        # How often to export z snapshots?
        print_progress_steps=100,
        random_seed=1000
):
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    assert start_img_id % uni_batch == 0, 'start_img-id shall be divisible by uni_batch'
    assert uni_batch % minibatch == 0, 'uni_bacth shall be divisible by minibatch'

    with tf.device('/gpu:0'):
        network_pkl = misc.locate_network_pkl(run_id, snapshot)
        print('Loading network from "%s"...' % network_pkl)
        G, D, Gs = misc.load_network_pkl(run_id, snapshot)

    print('Setting up result dir...')
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)

    if reals_np is None:
        training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)
        if start_img_id != 0:
            for _ in range(int(start_img_id / uni_batch)):
                _, _ = training_set.get_minibatch_np(minibatch_size=uni_batch)

    if num_images is None:
        num_images = uni_batch

    random_state = np.random.RandomState(random_seed)
    print('...... Start random sampling ......')
    if reals_np is None:
        reals, labels = training_set.get_minibatch_np(minibatch_size=uni_batch)
    else:
        reals = reals_np[start_img_id:(start_img_id + uni_batch)]
        labels = labels_np[start_img_id:(start_img_id + uni_batch)]

    # normalize reals
    reals = np.array(reals / 127.5 - 1, dtype=np.float32)
    reals = reals[0:num_images]
    labels = labels[0:num_images]

    start = time.time()
    z_samples = random_state.randn(num_sampling, 1, *Gs.input_shape[1:]).astype(np.float32)
    z_samples = np.repeat(z_samples, repeats=reals.shape[0], axis=1)
    error_log = np.zeros(z_samples.shape[0], dtype=np.float32)

    for ind, z in enumerate(z_samples):
        fakes = Gs.run(z, labels, minibatch_size=16)
        error_log[ind] = np.mean(np.square(reals - fakes))
    sort_ind = np.argsort(error_log, axis=0)
    sorted_error_log = np.take_along_axis(error_log, sort_ind, axis=0)
    sorted_z_samples = np.take_along_axis(z_samples, np.expand_dims(sort_ind, axis=[1, 2]), axis=0)
    z_optimals = np.repeat(sorted_z_samples[0, 0:1], repeats=minibatch, axis=0)

    savemat(os.path.join(result_subdir, 'z_sampling.mat'), {'z_sampling': z_optimals[0]})
    fake_images = Gs.run(z_optimals, labels, minibatch_size=minibatch)
    misc.save_image_grid(fake_images, os.path.join(result_subdir, 'fake_z_sampling.png'),
                         drange=drange_net)

    print('lowest initial error')
    print(sorted_error_log[0])
    print('Finish sampling method in %.1f sec' % (time.time() - start))

    plt.figure()
    plt.hist(error_log, density=True, bins=np.arange(0.05, 0.20, 0.005))
    plt.xlabel('Loss')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(result_subdir, 'error_hist.png'))
    plt.close()

    if reals_np is None:
        training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)
        if start_img_id != 0:
            for _ in range(int(start_img_id / minibatch)):
                _, _ = training_set.get_minibatch_np(minibatch_size=minibatch)
    num_trials = np.ceil(num_images / minibatch).astype(np.int32)
    loss_history = np.zeros((num_trials, total_step), dtype=np.float32)
    for index in range(num_trials):
        print('...... trial %d/%d ......' % (index + 1, num_trials))
        if reals_np is None:
            reals, labels = training_set.get_minibatch_np(minibatch_size=minibatch)
        else:
            reals = reals_np[(start_img_id + index * minibatch):(start_img_id + (index + 1) * minibatch)]
            labels = labels_np[(start_img_id + index * minibatch):(start_img_id + (index + 1) * minibatch)]
        # normalize reals
        reals = np.array(reals / 127.5 - 1, dtype=np.float32)

        misc.save_image_grid(reals, os.path.join(result_subdir, 'reals_trial%d.png' % index), drange=drange_net)

        print('Building TensorFlow graph...')
        with tf.name_scope('Inputs'):
            lrate_in = tf.placeholder(tf.float32, name='lrate_in', shape=[])
            latent_in = tf.Variable(initial_value=z_optimals,
                                    trainable=True,
                                    shape=z_optimals.shape,
                                    name='latent_in',
                                    dtype=tf.float32)
        z_opt = tf.train.AdamOptimizer(learning_rate=lrate_in)
        with tf.name_scope('GPU0'), tf.device('/gpu:0'):
            with tf.name_scope('z_loss'):
                cal_loss_op = tfutil.call_func_by_name(Gs=Gs,
                                                       D=D,
                                                       reals=reals,
                                                       labels=labels,
                                                       latents=latent_in,
                                                       **config.z_loss)
            # with tf.name_scope('z_loss'):
            #     fake_images_out = Gs_gpu.get_output_for(latent_in, labels, is_training=False)
            #     with tf.name_scope('MSE'):
            #         mse = tf.reduce_mean(tf.square(fake_images_out - reals))
            grad_and_var = z_opt.compute_gradients(cal_loss_op, [latent_in])
            z_op = z_opt.apply_gradients(grad_and_var)
            # latent_in = (latent_in - tf.math.reduce_mean(latent_in)) / (tf.math.reduce_std(latent_in) + 1e-8)

        print('Training...')
        tfutil.run(latent_in.initializer)
        tfutil.run(tf.variables_initializer(z_opt.variables()))

        # savemat(os.path.join(result_subdir, 'z%06d_trial%d.mat' % (0, index)), {'z': tfutil.run(latent_in)})
        # fake_images = Gs.run(tfutil.run(latent_in), labels, minibatch_size=minibatch)
        # misc.save_image_grid(fake_images, os.path.join(result_subdir, 'fakes%06d_trial%d.png' % (0, index)),
        #                      drange=drange_net)

        count = 0
        for cur_step in range(total_step):
            decayed_learning_rate = initial_learning_rate * decay_rate ** (cur_step // decay_steps)
            z_loss, _, _ = tfutil.run([cal_loss_op, grad_and_var, z_op],
                                      {lrate_in: max(decayed_learning_rate, lowest_learning_rate)})
            loss_history[index, cur_step] = z_loss
            # print('Step %-8d loss: %-8.5f' % (cur_step + 1, z_loss))
            if cur_step > 0 and loss_history[index, cur_step - 1] - loss_history[index, cur_step] < 0.00001:
                count = count + 1
            else:
                count = 0
            # tfutil.autosummary('Progress/learning_rate', decayed_learning_rate)
            cur_step += 1
            # tfutil.save_summaries(summary_log, cur_step)
            done = (cur_step == total_step)
            if cur_step % print_progress_steps == 0 or done:
                print('step %-8d learning_rate %-8e loss %-8.5f' % (cur_step, decayed_learning_rate, z_loss))
            # Save snapshots.
            if cur_step % image_snapshot_steps == 0:
                fake_images = Gs.run(tfutil.run(latent_in), labels, minibatch_size=minibatch)
                misc.save_image_grid(fake_images,
                                     os.path.join(result_subdir, 'fakes%06d_trial%d.png' % (cur_step, index)),
                                     drange=drange_net)
            if cur_step % z_snapshot_steps == 0:
                savemat(os.path.join(result_subdir, 'z%06d_trial%d.mat' % (cur_step, index))
                        , {'z': tfutil.run(latent_in)})
            if count > 5 and cur_step > mini_step:
                break

        # save final results.
        fake_images = Gs.run(tfutil.run(latent_in), labels, minibatch_size=minibatch)
        misc.save_image_grid(fake_images,
                             os.path.join(result_subdir, 'fakes_final_trial%d.png' % index),
                             drange=drange_net)
        savemat(os.path.join(result_subdir, 'z_final_trial%d.mat' % index), {'z': tfutil.run(latent_in)})
        # summary_log.close()

    plt.figure()
    for tt in range(num_trials):
        plt.plot(loss_history[tt], label='trial%d' % tt)
    plt.xlabel('Step #')
    plt.ylabel('MSE')
    # plt.yscale('log')
    plt.legend(loc='best')
    # plt.tight_layout()
    plt.savefig(os.path.join(result_subdir, 'history.png'))
    plt.close()
    savemat(os.path.join(result_subdir, 'loss.mat'), {'loss': loss_history})
    open(os.path.join(result_subdir, '_training-done.txt'), 'wt').close()

    open(os.path.join(result_subdir, '_training-done.txt'), 'wt').close()


def extract_pore_structure():
    from skimage import morphology
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    print('Setting up result dir...')
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)
    # load dataset
    data = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)
    imgs, labels = data.get_minibatch_np(minibatch_size=1725)  # (1725, 1, 1024, 1024)
    imgs = imgs[0:256, 0, 0:256, 0:256]  # (1725, 1024, 1024)
    # segmentation
    pore = np.zeros_like(imgs)
    pore[np.logical_and(imgs >= 0, imgs <= 97)] = 1
    # porous_solid = np.zeros_like(imgs)
    # porous_solid[np.logical_and(imgs >= 98, imgs <= 144)] = 1
    # solid = np.zeros_like(imgs)
    # solid[np.logical_and(imgs >= 145, imgs <= 255)] = 1
    #
    # label
    pore_label, num_pore = morphology.label(pore, background=0, return_num=True, connectivity=pore.ndim)
    # # porous_solid_label, num_porous_solid = morphology.label(porous_solid, background=0, return_num=True,
    # #                                                         connectivity=porous_solid.ndim)
    # # solid_label, num_solid = morphology.label(solid, background=0, return_num=True, connectivity=solid.ndim)
    # # print('num_pore: %d, num_porous_solid:%d, num_soild: %d' % (num_pore, num_porous_solid, num_solid))
    # print('num_pore: %d' % num_pore)
    #
    # i = 1
    # while i <= np.max(pore_label):
    #     mask = np.zeros_like(pore_label)
    #     mask[pore_label == i] = 1
    #     if np.sum(mask) < 8000:
    #         num_pore = num_pore - 1
    #         pore_label[pore_label == i] = 0
    #         pore_label[pore_label > i] = pore_label[pore_label > i] - 1
    #         continue
    #     i = i + 1
    #
    # print('num_pore: %d' % num_pore)
    #
    # # plot
    # x = np.arange(1024)
    # y = np.arange(1024)
    # z = np.arange(1024)
    # xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
    # plt.figure(figsize=(10, 10))
    # plt.scatter(xg, yg, zg, c=pore_label)
    # plt.colorbar()
    # plt.savefig(result_subdir + 'pore.png')
    # plt.close()

    # # plot
    # pore = pore[0:512, 0:512, 0:512]
    # x = np.arange(512)
    # y = np.arange(512)
    # z = np.arange(512)
    # xg, yg, zg = np.meshgrid(x, y, z, indexing='ij')
    # xg[pore == 0] = 0
    # yg[pore == 0] = 0
    # zg[pore == 0] = 0
    #
    # fig = plt.figure(figsize=(10, 10))
    # ax = Axes3D(fig)
    #
    # def init():
    #     ax.scatter(xg, yg, zg, c=pore)
    #     return fig,
    #
    # def animate(i):
    #     ax.view_init(elev=10., azim=i)
    #     return fig,
    #
    # # Animate
    # anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                                frames=360, interval=20, blit=True)
    # # Save
    # anim.save(result_subdir+'/pore.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    bc = np.bincount(pore_label.flatten(), minlength=num_pore + 1)

    # label = np.arange(num_pore + 1)
    # label[bc < 1000] = 0
    cube = np.zeros_like(pore_label)
    cube[pore_label == np.argmax(bc[1:]) + 1] = 1
    print(bc[:10])
    print(np.argmax(bc[1:]) + 1)
    print(np.max(bc[1:]))
    # for item in label:
    #     if item:
    #         cube[pore_label == item] = 1
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    def init():
        ax.voxels(cube, facecolors='#1f77b430', edgecolor='#1f77b430')
        return fig,

    # plt.savefig(result_subdir + '/pore.png')

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    # Animate
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=360, interval=20, blit=True)
    # Save
    anim.save(result_subdir + '/pore.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

    # plt.figure(figsize=(10, 10))
    # plt.scatter(xg, yg, zg, c=porous_solid_label)
    # plt.colorbar()
    # plt.savefig(result_subdir + 'porous_solid.png')
    # plt.close()
    #
    # plt.figure(figsize=(10, 10))
    # plt.scatter(xg, yg, zg, c=solid_label)
    # plt.colorbar()
    # plt.savefig(result_subdir + 'solid.png')
    # plt.close()
