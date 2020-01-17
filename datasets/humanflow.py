import os.path
import glob
from .listdataset import ListDataset
from .util import split2list


def make_dataset(dir, phase='train'):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    images = []
    for flow_map in sorted(glob.glob(os.path.join(dir, phase+'/*/flow/*.flo'))):
        flow_map = os.path.relpath(flow_map, dir)
        img1 = flow_map.replace('/flow/', '/composition/')
        img1 = img1.replace('.flo', '.png')
        img2 = img1[:-9] + str(int(img1.split('/')[-1][:-4])+1).zfill(5) + '.png'

        if int(img1.split('/')[-1][:-4]) % 10 == 9:
            continue

        if not (os.path.isfile(os.path.join(dir,img1)) and os.path.isfile(os.path.join(dir,img2))):
            continue

        images.append([[img1,img2],flow_map])

    return images


def humanflow(root, transform=None, target_transform=None,
                  co_transform=None, split=None):
    train_list = make_dataset(root, 'train')
    test_list = make_dataset(root, 'test')
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list, transform, target_transform)

    return train_dataset, test_dataset
