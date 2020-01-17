import argparse
import os
import glob
from tqdm import tqdm
import cv2
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imsave


parser = argparse.ArgumentParser(description='Test Optical Flow',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--pred-dir', dest='pred_dir', type=str, default=None,
                    help='path to prediction folder')
parser.add_argument('--output-dir', dest='output_dir', type=str, default=None,
                    help='path to save visualizations')

def main():
    global args
    args = parser.parse_args()
    # test_list = make_dataset(args.data)
    test_list = make_real_dataset(args.data)

    # dirs = ['flownet2/inference/run.epoch-0-flow-field', 'LDOF', 'pcaflow',
    #         'EpicFlow', 'spynet', 'spynet_mhf', 'pwc', 'pwc_mhf']
    dirs = ['flownet2_real', 'LDOF_real', 'pcaflow_real', 'EpicFlow_real', 'flownet2s_real', 'spynet_real', 'spynet_mhf_real', 'pwc_real', 'pwc_mhf_real']

    for i, (img1path, img2path, flowpath) in enumerate(tqdm(test_list)):
        img1 = imread(img1path, mode='RGB')
        img2 = imread(img2path, mode='RGB')
        if flowpath is not None:
            gtflow = flow2rgb(load_flo(flowpath))

        predflows = {}
        pathexists = True
        for d in dirs:
            if flowpath is not None:
                fpath = flowpath.replace(args.data, args.pred_dir)
                fpath = fpath.replace('/test/', '/'+d+'/')
                fpath = fpath.replace('/flow/', '/')
            else:
                fpath = img1path.replace(args.data, args.pred_dir)
                fpath = fpath.replace('.png', '.flo')
                fpath = fpath.replace('/flow_evaluation/', '/flow_evaluation/'+d+'/')
                if os.path.isfile(fpath):
                    predflows[d] = flow2rgb(load_flo(fpath))
                else:
                    pathexists = False

        if not pathexists:
            continue
        if flowpath is not None:
            toprow = np.hstack((img1[:,:,:3], predflows[dirs[0]], predflows[dirs[1]], predflows[dirs[2]], predflows[dirs[3]]))
            bottomrow = np.hstack((gtflow, predflows[dirs[4]], predflows[dirs[5]], predflows[dirs[6]], predflows[dirs[7]]))

        else:
            toprow = np.hstack((img1[:,:,:3], predflows[dirs[0]], predflows[dirs[1]], predflows[dirs[2]], predflows[dirs[3]]))
            bottomrow = np.hstack((predflows[dirs[4]], predflows[dirs[5]], predflows[dirs[6]], predflows[dirs[7]], predflows[dirs[8]]))

        viz_im = np.vstack((toprow, bottomrow))
        save_path = fpath.replace(args.pred_dir, args.output_dir).replace(dirs[-1]+'/', '').replace('.flo', '.png')
        os.system('mkdir -p '+os.path.dirname(save_path))
        imsave(save_path, viz_im)



def flow2rgb(flow_map, max_value=None):
    flow_map_np = flow_map.transpose(2,0,1)
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    rgb_flow = rgb_map.clip(0,1)
    rgb_flow = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
    return rgb_flow



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

def make_real_dataset(dir):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    images = []
    for img1 in sorted( glob.glob(os.path.join(dir, '*/*1.png')) ):
        img2 = img1[:-9] + str(int(img1.split('/')[-1][:-4])+1).zfill(5) + '.png'

        if int(img1.split('/')[-1][:-4]) % 10 == 9:
            continue

        if int(img1.split('/')[-1][:-4]) < 90:
            continue

        if not (os.path.isfile(os.path.join(dir,img1)) and os.path.isfile(os.path.join(dir,img2))):
            continue

        images.append([img1, img2, None])

    return images


def make_dataset(dir, phase='test'):
    '''Will search for triplets that go by the pattern '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    images = []
    for flow_map in sorted(glob.glob(os.path.join(dir, phase+'/*/flow/*.flo'))):
        #flow_map = os.path.relpath(flow_map, dir)
        img1 = flow_map.replace('/flow/', '/composition/')
        img1 = img1.replace('.flo', '.png')
        img2 = img1[:-9] + str(int(img1.split('/')[-1][:-4])+1).zfill(5) + '.png'

        #seg_mask = flow_map.replace('/flow/', '/segm_EXR/')
        #seg_mask = seg_mask.replace('.flo', '.exr')

        #pred_flow = flow_map.replace(args.data, args.pred_dir).replace('/test/', '/').replace('/flow/','/')
        if int(img1.split('/')[-1][:-4]) % 10 == 9:
            continue

        if not (os.path.isfile(os.path.join(dir,img1)) and os.path.isfile(os.path.join(dir,img2))):
            continue

        images.append([img1, img2, flow_map])

    return images

if __name__ == '__main__':
    main()
