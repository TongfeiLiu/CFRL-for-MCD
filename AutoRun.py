import os

# for ps in range(3, 23, 2):
#     path = './vision/ps_{}/'.format(ps)
#     if not os.path.exists(path):
#         os.makedirs(path)
#     os.system('python train.py --patch_size={} --vision_path={}'.format(ps, path))

for tps in range(3, 11, 2):
    pth = './vision/test_ps{}/'.format(tps)
    if not os.path.exists(pth):
        os.makedirs(pth)
    for iter in range(1, 10, 1):
        print('-------- test_ps-{}: iter-{} ---------\n'.format(tps, iter))
        path = './vision/test_ps{}/iter_{}/'.format(tps, iter)
        if not os.path.exists(path):
            os.makedirs(path)
        os.system('python train.py --test_ps={} --vision_path={}'.format(tps, path))
