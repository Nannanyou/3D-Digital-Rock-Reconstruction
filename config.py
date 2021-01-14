# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# ----------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".


class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)

    def __getattr__(self, name): return self[name]

    def __setattr__(self, name, value): self[name] = value

    def __delattr__(self, name): del self[name]


# ----------------------------------------------------------------------------
# Paths.

data_dir = 'datasets'
result_dir = 'results_new'

# ----------------------------------------------------------------------------
# TensorFlow options.
tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()  # Environment variables, set by the main program in train.py.

tf_config['graph_options.place_pruned_graph'] = True
# False (default) = Check that all ops are available on the designated device.
# True = Skip the check for ops that are not used.

# tf_config['gpu_options.allow_growth'] = False
# False (default) = Allocate all GPU memory at the beginning.
# True = Allocate only as much GPU memory as needed.

# env.CUDA_VISIBLE_DEVICES = '0'
# Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL = '1'
# 0 (default) = Print all available debug info from TensorFlow.
# 1 = Print warnings and errors, but disable debug info.

# ----------------------------------------------------------------------------
# Official training configs, targeted mainly for CelebA-HQ.
# To run, comment/uncomment the lines as appropriate and launch train.py.

desc = 'pgan'  # Description string included in result subdir name.
random_seed = 1000  # Global random seed.
dataset = EasyDict()  # Options for dataset.load_dataset().
train = EasyDict(func='train.train_progressive_gan')  # Options for main training func.
G = EasyDict(func='networks.G_paper')  # Options for generator network.
D = EasyDict(func='networks.D_paper')  # Options for discriminator network.
G_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)  # Options for generator optimizer.
D_opt = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)  # Options for discriminator optimizer.
G_loss = EasyDict(func='loss.G_wgan_acgan')  # Options for generator loss.
D_loss = EasyDict(func='loss.D_wgangp_acgan')  # Options for discriminator loss.
sched = EasyDict()  # Options for train.TrainingSchedule.
grid = EasyDict(size='1080p', layout='random')  # Options for train.setup_snapshot_image_grid().

# desc += '-latent256'
# G.latent_size = 256

desc += '-latent512'
G.latent_size = 512

# desc += '-linearup_convk2'
# G.fused_scale = False
# G.up_scale_method = 'linear'

desc += '-fusedup_convk2_1conv'
G.fused_scale = True
G.up_scale_method = 'nearest'

desc += '-1conv_fuseddown_convk3'

# desc += '-bicubicup'
# G.fused_scale = False
# G.up_scale_method = 'bicubic'


# Dataset (choose one).
desc += '-case11_extended'
dataset = EasyDict(tfrecord_dir='carbonate_case11_extended')

# desc += '-sandstone'
# dataset = EasyDict(tfrecord_dir='sandstone')

# desc += '-rock'
# dataset = EasyDict(tfrecord_dir='rock')
# G.initial_resolution = 6

train.mirror_augment = True

# # resume
# train.resume_run_id = 6
# # Run ID or network pkl to resume training from, None = start from scratch.
# train.resume_kimg = 12000
# # Assumed training progress at the beginning. Affects reporting and training schedule.
# train.resume_time = 37 * 60 * 60 + 12 * 60

# desc += '-celeba'      dataset = EasyDict(tfrecord_dir='celeba')
# train.mirror_augment = True
# desc += '-cifar10'     dataset = EasyDict(tfrecord_dir='cifar10')
# desc += '-cifar100'    dataset = EasyDict(tfrecord_dir='cifar100')
# desc += '-svhn'        dataset = EasyDict(tfrecord_dir='svhn')
# desc += '-mnist'       dataset = EasyDict(tfrecord_dir='mnist')
# desc += '-mnistrgb'    dataset = EasyDict(tfrecord_dir='mnistrgb')
# desc += '-syn1024rgb'      dataset = EasyDict(class_name='dataset.SyntheticDataset', resolution=1024, num_channels=3)
# desc += '-lsun-airplane'   dataset = EasyDict(tfrecord_dir='lsun-airplane-100k')
# train.mirror_augment = True
# desc += '-lsun-bedroom'    dataset = EasyDict(tfrecord_dir='lsun-bedroom-100k')
# train.mirror_augment = True
# desc += '-lsun-bicycle'    dataset = EasyDict(tfrecord_dir='lsun-bicycle-100k')
# train.mirror_augment = True
# desc += '-lsun-bird'       dataset = EasyDict(tfrecord_dir='lsun-bird-100k')
# train.mirror_augment = True
# desc += '-lsun-boat'       dataset = EasyDict(tfrecord_dir='lsun-boat-100k')
# train.mirror_augment = True
# desc += '-lsun-bottle'     dataset = EasyDict(tfrecord_dir='lsun-bottle-100k')
# train.mirror_augment = True
# desc += '-lsun-bridge'     dataset = EasyDict(tfrecord_dir='lsun-bridge-100k')
# train.mirror_augment = True
# desc += '-lsun-bus'        dataset = EasyDict(tfrecord_dir='lsun-bus-100k')
# train.mirror_augment = True
# desc += '-lsun-car'        dataset = EasyDict(tfrecord_dir='lsun-car-100k')
# train.mirror_augment = True
# desc += '-lsun-cat'        dataset = EasyDict(tfrecord_dir='lsun-cat-100k')
# train.mirror_augment = True
# desc += '-lsun-chair'      dataset = EasyDict(tfrecord_dir='lsun-chair-100k')
# train.mirror_augment = True
# desc += '-lsun-churchoutdoor'  dataset = EasyDict(tfrecord_dir='lsun-churchoutdoor-100k')
# train.mirror_augment = True
# desc += '-lsun-classroom'      dataset = EasyDict(tfrecord_dir='lsun-classroom-100k')
# train.mirror_augment = True
# desc += '-lsun-conferenceroom' dataset = EasyDict(tfrecord_dir='lsun-conferenceroom-100k')
# train.mirror_augment = True
# desc += '-lsun-cow'            dataset = EasyDict(tfrecord_dir='lsun-cow-100k')
# train.mirror_augment = True
# desc += '-lsun-diningroom'     dataset = EasyDict(tfrecord_dir='lsun-diningroom-100k')
# train.mirror_augment = True
# desc += '-lsun-diningtable'    dataset = EasyDict(tfrecord_dir='lsun-diningtable-100k')
# train.mirror_augment = True
# desc += '-lsun-dog'            dataset = EasyDict(tfrecord_dir='lsun-dog-100k')
# train.mirror_augment = True
# desc += '-lsun-horse'          dataset = EasyDict(tfrecord_dir='lsun-horse-100k')
# train.mirror_augment = True
# desc += '-lsun-kitchen'        dataset = EasyDict(tfrecord_dir='lsun-kitchen-100k')
# train.mirror_augment = True
# desc += '-lsun-livingroom'     dataset = EasyDict(tfrecord_dir='lsun-livingroom-100k')
# train.mirror_augment = True
# desc += '-lsun-motorbike'      dataset = EasyDict(tfrecord_dir='lsun-motorbike-100k')
# train.mirror_augment = True
# desc += '-lsun-person'         dataset = EasyDict(tfrecord_dir='lsun-person-100k')
# train.mirror_augment = True
# desc += '-lsun-pottedplant'    dataset = EasyDict(tfrecord_dir='lsun-pottedplant-100k')
# train.mirror_augment = True
# desc += '-lsun-restaurant'     dataset = EasyDict(tfrecord_dir='lsun-restaurant-100k')
# train.mirror_augment = True
# desc += '-lsun-sheep'          dataset = EasyDict(tfrecord_dir='lsun-sheep-100k')
# train.mirror_augment = True
# desc += '-lsun-sofa'           dataset = EasyDict(tfrecord_dir='lsun-sofa-100k')
# train.mirror_augment = True
# desc += '-lsun-tower'          dataset = EasyDict(tfrecord_dir='lsun-tower-100k')
# train.mirror_augment = True
# desc += '-lsun-train'          dataset = EasyDict(tfrecord_dir='lsun-train-100k')
# train.mirror_augment = True
# desc += '-lsun-tvmonitor'      dataset = EasyDict(tfrecord_dir='lsun-tvmonitor-100k')
# train.mirror_augment = True

# Conditioning & snapshot options.
# desc += '-cond_ce'
# dataset.max_label_size = 'full'  # conditioned on full label
# D.skipSoftmax = False  # the output of the Discriminator would be score, (percentage of phase 1, phase 2 and phase 3)
# G_loss.label_loss = 'ce'
# D_loss.label_loss = 'ce'
# G_loss.cond_weight = 1
# D_loss.cond_weight = 1

# desc += '-cond_mse'
# dataset.max_label_size = 'full'  # conditioned on full label
# D.skipSoftmax = True  # the output of the Discriminator would be score, (percentage of phase 1, phase 2 and phase 3)
# G_loss.label_loss = 'mse'
# D_loss.label_loss = 'mse'
# desc += '-cond1'
# dataset.max_label_size = 1 # conditioned on first component of the label
# desc += '-g4k'
# grid.size = '4k'
# desc += '-grpc'
# grid.layout = 'row_per_class'

# normalize latent parameters
desc += '-normLatent_T'
G.normalize_latents = True
# desc += '-normLatent_F'
# G.normalize_latents = False

# Config presets (choose one).
# desc += '-preset-v1-1gpu'
# num_gpus = 1
# D.mbstd_group_size = 16
# sched.minibatch_base = 16
# sched.minibatch_dict = {256: 14, 512: 6, 1024: 3}
# sched.lod_training_kimg = 800
# sched.lod_transition_kimg = 800
# train.total_kimg = 19000
# desc += '-preset-v2-1gpu'
# num_gpus = 1
# sched.minibatch_base = 4
# sched.minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}
# sched.G_lrate_dict = {1024: 0.0015}
# sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)
# train.total_kimg = 12000
# desc += '-preset-v2-2gpus'
# num_gpus = 2
# sched.minibatch_base = 8
# sched.minibatch_dict = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
# sched.G_lrate_dict = {512: 0.0015, 1024: 0.002}
# sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)
# train.total_kimg = 12000

# desc += '-preset-v2-4gpus'
# num_gpus = 4
# sched.minibatch_base = 16
# sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
# sched.G_lrate_dict = {256: 0.0015, 512: 0.002, 1024: 0.003}
# sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)
# train.total_kimg = 12000

desc += '-preset-v2-8gpus'
num_gpus = 8
sched.minibatch_base = 32
sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}
sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)
sched.lod_training_kimg = 600
sched.lod_transition_kimg = 600
train.total_kimg = 12000

# # Numerical precision (choose one).
# desc += '-fp32'
# sched.max_minibatch_per_gpu = {256: 16, 512: 8, 1024: 4}
# desc += '-fp16'
# G.dtype = 'float16'
# D.dtype = 'float16'
# G.pixelnorm_epsilon=1e-4
# G_opt.use_loss_scaling = True
# D_opt.use_loss_scaling = True
# sched.max_minibatch_per_gpu = {512: 16, 1024: 8}

# Disable individual features.
# desc += '-nogrowing'
# sched.lod_initial_resolution = 1024
# sched.lod_training_kimg = 0
# sched.lod_transition_kimg = 0
# train.total_kimg = 10000
# desc += '-nopixelnorm'
# G.use_pixelnorm = False
# desc += '-nowscale'
# G.use_wscale = False
# D.use_wscale = False
# desc += '-noleakyrelu'
# G.use_leakyrelu = False
# desc += '-nosmoothing'
# train.G_smoothing = 0.0
# desc += '-norepeat'
# train.minibatch_repeats = 1
# desc += '-noreset'
# train.reset_opt_for_new_lod = False

# Special modes.
# desc += '-BENCHMARK'
# sched.lod_initial_resolution = 4
# sched.lod_training_kimg = 3
# sched.lod_transition_kimg = 3
# train.total_kimg = (8*2+1)*3
# sched.tick_kimg_base = 1
# sched.tick_kimg_dict = {}
# train.image_snapshot_ticks = 1000
# train.network_snapshot_ticks = 1000
# desc += '-BENCHMARK0'
# sched.lod_initial_resolution = 1024
# train.total_kimg = 10
# sched.tick_kimg_base = 1
# sched.tick_kimg_dict = {}
# train.image_snapshot_ticks = 1000
# train.network_snapshot_ticks = 1000
# desc += '-VERBOSE'
# sched.tick_kimg_base = 1
# sched.tick_kimg_dict = {}
# train.image_snapshot_ticks = 1
# train.network_snapshot_ticks = 100
# desc += '-GRAPH'
# train.save_tf_graph = True
desc += '-HIST'
train.save_weight_histograms = True

# ----------------------------------------------------------------------------
# Utility scripts.
# To run, uncomment the appropriate line and launch train.py.
#
# train = EasyDict(func='util_scripts.generate_fake_labelled_images', run_id=59, num_pngs=10, label=[0, 1],
#                  random_seed=1000)
# num_gpus = 1
# desc = 'fake-images-id' + str(train.run_id) + '-' + str(train.label[0]) + '-' + str(train.label[1])
# train = EasyDict(func='util_scripts.generate_fake_images', run_id=241, num_pngs=40)
# num_gpus = 1
# desc = 'fake-images-' + str(train.run_id)
# train = EasyDict(func='util_scripts.generate_specified_images', run_id=72, grid_size=[1, 1])
# num_gpus = 1
# desc = 'specified-images-' + str(train.run_id)
dataset = EasyDict(tfrecord_dir='carbonate_in_order')
dataset.shuffle_mb = 0
train = EasyDict(func='util_scripts.generate_images_from_manipulated_noise', run_id=241, grid_size=[1, 1],
                 start=0, window_size=9, end=1720)
num_gpus = 1
desc = 'Interpolated-image_'+str(train.start)+'_'+str(train.window_size)+'-' + str(train.run_id)

# train = EasyDict(func='util_scripts.interpolation_between_z', run_id=241,
#                  start=0, window_size=9, end=1720)
# num_gpus = 1
# desc = 'z-interpolation_'+str(train.start)+'_'+str(train.window_size)+'-' + str(train.run_id)

# dataset = EasyDict(tfrecord_dir='carbonate_in_order')
# dataset.shuffle_mb = 0
# train = EasyDict(func='util_scripts.interpolation_between_images', grid_size=[1, 1],
#                  start=0, window_size=9, end=1720)
# num_gpus = 1
# desc = 'Image-interpolation_'+str(train.start)+'_'+str(train.window_size)+'_'+str(train.end)

# dataset = EasyDict(tfrecord_dir='carbonate_in_order')
# dataset.shuffle_mb = 0
# train = EasyDict(func='util_scripts.generate_thirdD_real_images', grid_size=[1, 1])
# num_gpus = 1
# desc = 'real-images'
# train = EasyDict(func='util_scripts.generate_fake_images', run_id=23, grid_size=[15,8], num_pngs=10, image_shrink=4)
# desc = 'fake-grids-' + str(train.run_id)
# train = EasyDict(func='util_scripts.generate_interpolation_video', random_seed=1000,
#                  run_id=6, grid_size=[1, 1], duration_sec=32.0, smoothing_sec=8.0, mp4_fps=8, transit_sec=8)
# num_gpus = 1
# desc = 'gaussian-interpolation-video-' + str(train.run_id)
# train = EasyDict(func='util_scripts.generate_interpolation_image',
#                  run_id=72, start=0, window_size=6, end=1723)
# num_gpus = 1
# desc = 'randomly-generated-image-' + str(train.run_id)
# train = EasyDict(func='util_scripts.generate_training_video', run_id=0, duration_sec=20.0)
# num_gpus = 1
# desc = 'training-video-' + str(train.run_id)

# train = EasyDict(func='util_scripts.evaluate_metrics', run_id=0, log='metric-swd-16k.txt',
#                  metrics=['swd'], num_images=16384, real_passes=2)
# num_gpus = 1
# desc = train.log.split('.')[0] + '-' + str(train.run_id)

# train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-fid-10k.txt',
#                  metrics=['fid'], num_images=10000, real_passes=1)
# num_gpus = 1
# desc = train.log.split('.')[0] + '-' + str(train.run_id)
# train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-fid-50k.txt',
#                  metrics=['fid'], num_images=50000, real_passes=1)
# num_gpus = 1
# desc = train.log.split('.')[0] + '-' + str(train.run_id)

# train = EasyDict(func='util_scripts.evaluate_metrics', run_id=23, log='metric-is-50k.txt',
#                  metrics=['is'], num_images=50000, real_passes=1)
# num_gpus = 1
# desc = train.log.split('.')[0] + '-' + str(train.run_id)

# train = EasyDict(func='util_scripts.evaluate_metrics', run_id=72, log='metric-msssim-2k.txt',
#                  metrics=['msssim'], num_images=2000, real_passes=1)
# num_gpus = 1
# desc = train.log.split('.')[0] + '-' + str(train.run_id)
# dataset = EasyDict(tfrecord_dir='carbonate_in_order')
# dataset.shuffle_mb = 0
# z_loss = EasyDict(func='loss.z_loss')
# train = EasyDict(func='util_scripts.invert_gan',
#                  run_id=266,
#                  initial_learning_rate=0.01,
#                  lowest_learning_rate=0.005,
#                  num_sampling=2000,
#                  num_images=None,
#                  start_img_id=0,
#                  # uni_batch=512,
#                  minibatch=16,
#                  total_step=12000,
#                  print_progress_steps=1000,
#                  decay_rate=0.8,
#                  decay_steps=1000,
#                  random_seed=1000)
# num_gpus = 1
# desc = 'invert-gan-' + str(train.run_id)
# + '-imgID' + str(train.start_img_id)
# + '-resumeID' + str(train.resume_invert_id)
# dataset = EasyDict(tfrecord_dir='carbonate_in_order')
# dataset.shuffle_mb = 0
# train = EasyDict(func='util_scripts.uni_invert_gan',
#                  run_id=72,
#                  resume_invert_id=None,
#                  num_sampling=1000,
#                  num_images=1725,
#                  minibatch=1725,
#                  random_seed=1000)
# num_gpus = 1
# desc = 'uni-invert-gan-NNid' + str(train.run_id) \
#        + '-img' + str(train.num_images)
# + '-trial' + str(train.no_initial_z) \
# + '-seed' + str(train.random_seed) \
# + '-numsample' + str(train.num_sampling)
#
# dataset = EasyDict(tfrecord_dir='carbonate')
# train = EasyDict(func='util_scripts.SA_gan',
#                  run_id=42,
#                  num_sampling=1000,
#                  num_images=1,
#                  num_trials=4,
#                  minibatch=1,
#                  snapshot=None,
#                  P1=0.8,
#                  P2=0.03,
#                  Iter=100,
#                  Mo=2560,
#                  Maxtime=256000,
#                  beta=1,
#                  drange_net=[-1, 1],
#                  random_seed=1234,
#                  T_func='exp')
# num_gpus = 1
# desc = 'SA-gan-standard-' + train.T_func + '-' + str(train.run_id)

# train = EasyDict(func='util_scripts.pkl_to_pth', run_id=72)
# num_gpus = 1
# desc = 'pkl-to-pth-' + '-' + str(train.run_id)


# dataset = EasyDict(tfrecord_dir='carbonate_in_order')
# dataset.shuffle_mb = 0
# train = EasyDict(func='util_scripts.extract_pore_structure')
# num_gpus = 1
# desc = '3D-structure-'+dataset.tfrecord_dir
# ----------------------------------------------------------------------------
