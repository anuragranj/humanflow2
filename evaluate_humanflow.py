import argparse
import glob
import os
import numpy as np
import cv2

from scipy.ndimage import imread
from scipy.misc import imsave
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import models
from multiscaleloss import realEPE
import flow_transforms
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Test Optical Flow',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--pred-dir', dest='pred_dir', type=str, default=None,
                    help='path to prediction folder')
parser.add_argument('--save-name', dest='save_name', type=str, default=None,
                    help='Name for saving results')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args
    args = parser.parse_args()
    test_list = make_dataset(args.data)

    flow_epe = AverageMeter()
    avg_parts_epe = {}
    for bk in BODY_MAP.keys():
        avg_parts_epe[bk] = AverageMeter()


    for i, (pred_path, flow_path, seg_path) in enumerate(tqdm(test_list)):
        predflow = flow_transforms.ArrayToTensor()(load_flo(pred_path))
        gtflow = flow_transforms.ArrayToTensor()(load_flo(flow_path))
        segmask = flow_transforms.ArrayToTensor()(cv2.imread(seg_path))

        predflow_var = predflow.unsqueeze(0)
        gtflow_var = gtflow.unsqueeze(0)
        segmask_var = segmask.unsqueeze(0)

        predflow_var = predflow_var.to(device)
        gtflow_var = gtflow_var.to(device)
        segmask_var = segmask_var.to(device)

        # compute output
        epe = realEPE(predflow_var, gtflow_var)
        epe_parts = partsEPE(predflow_var, gtflow_var, segmask_var)
        #epe_parts.update((x, args.div_flow*y) for x, y in epe_parts.items() )

        # record EPE
        flow_epe.update(epe.item(), gtflow_var.size(0))
        for bk in avg_parts_epe:
            if epe_parts[bk].item() > 0:
                avg_parts_epe[bk].update(epe_parts[bk].item(), gtflow_var.size(0))

    epe_dict = {}
    for bk in BODY_MAP.keys():
        epe_dict[bk] = avg_parts_epe[bk].avg
    epe_dict['full_epe'] = flow_epe.avg
    np.save(os.path.join('results', args.save_name), epe_dict)


    print("Averge EPE",flow_epe.avg )

def partsEPE(output, gtflow, seg_mask):
    parts_epe_dict = {}
    for bk in BODY_MAP.keys():
        mask = seg_mask == BODY_MAP[bk]
        gt_partflow = mask.type_as(gtflow)[:,:2,:,:] * gtflow
        epe_part = realEPE(output, gt_partflow, sparse=True)
        parts_epe_dict[bk] = epe_part

    return parts_epe_dict


def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D

def make_dataset(dir, phase='test'):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    images = []
    for flow_map in sorted(glob.glob(os.path.join(dir, phase+'/*/flow/*.flo'))):
        #flow_map = os.path.relpath(flow_map, dir)
        img1 = flow_map.replace('/flow/', '/composition/')
        img1 = img1.replace('.flo', '.png')
        img2 = img1[:-9] + str(int(img1.split('/')[-1][:-4])+1).zfill(5) + '.png'

        seg_mask = flow_map.replace('/flow/', '/segm_EXR/')
        seg_mask = seg_mask.replace('.flo', '.exr')

        pred_flow = flow_map.replace(args.data, args.pred_dir).replace('/test/', '/').replace('/flow/','/')
        if int(img1.split('/')[-1][:-4]) % 10 == 9:
            continue

        if not (os.path.isfile(os.path.join(dir,img1)) and os.path.isfile(os.path.join(dir,img2))):
            continue

        images.append([pred_flow, flow_map, seg_mask])

    return images

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)

BODY_MAP = {'global': 1, 'head': 16, 'lIndex0': 23, 'lIndex1': 33, 'lIndex2': 43,
        'lMiddle0': 24, 'lMiddle1': 34, 'lMiddle2': 44, 'lPinky0': 25,
        'lPinky1': 35, 'lPinky2': 45, 'lRing0': 26, 'lRing1': 36, 'lRing2': 46,
        'lThumb0': 27, 'lThumb1': 37, 'lThumb2': 47, 'leftCalf': 5, 'leftFoot': 8,
        'leftForeArm': 19, 'leftHand': 21, 'leftShoulder': 14, 'leftThigh': 2, 'leftToes': 11,
        'leftUpperArm': 17, 'neck': 13, 'rIndex0': 28, 'rIndex1': 38, 'rIndex2': 48,
        'rMiddle0': 29, 'rMiddle1': 39, 'rMiddle2': 49, 'rPinky0': 30, 'rPinky1': 40,
        'rPinky2': 50, 'rRing0': 31, 'rRing1': 41, 'rRing2': 51, 'rThumb0': 32,
        'rThumb1': 42, 'rThumb2': 52, 'rightCalf': 6, 'rightFoot': 9, 'rightForeArm': 20,
        'rightHand': 22, 'rightShoulder': 15, 'rightThigh': 3, 'rightToes': 12, 'rightUpperArm': 18,
        'spine': 4, 'spine1': 7, 'spine2': 10}


if __name__ == '__main__':
    main()
