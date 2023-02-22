import os
import os.path as ops
import numpy as np
import cv2
import json
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as T


class LaneDataset(Dataset):
    def __init__(self, args, dataset_path, json_file_path, transform=None, data_aug=False):
        self.is_testing = ('test' in json_file_path) # 'val'
        self.num_class = args.num_class

        # define image pre-processor
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(args.vgg_mean, args.vgg_std)
        self.data_aug = data_aug

        # dataset parameters
        self.img_path = dataset_path
        self.transform = transform
        self.h_org = args.org_h
        self.w_org = args.org_w
        self.h_crop = args.crop_y
        # self.K = args.K

        # parameters related to service network
        self.h_net = args.resize_h
        self.w_net = args.resize_w
        # self.H_crop = homography_crop_resize([args.org_h, args.org_w], args.crop_y, [args.resize_h, args.resize_w])

        self._label_image_path, self._label_laneline_all, self._gt_class_label_all = self.init_dataset_3D(dataset_path, json_file_path)

    def __len__(self):
        return len(self._label_image_path)

    def __getitem__(self, idx):
        img_name = self._label_image_path[idx]

        # prepare binary segmentation label map
        label_map = np.zeros((self.h_org, self.w_org), dtype=np.int8)
        gt_lanes = self._label_laneline_all[idx]
        gt_labels = self._gt_class_label_all[idx]
        for i, coords in enumerate(gt_lanes):
            # skip the class label beyond consideration
            if gt_labels[i] <= 0 or gt_labels[i] > self.num_class:
                continue

            for j in range(len(coords) - 1):
                label_map = cv2.line(label_map, coords[j], coords[j+1], color=np.array(gt_labels[i]).item(), thickness=3)
        label_map = Image.fromarray(label_map)
        
        label_map = F.crop(label_map, self.h_crop, 0, self.h_org - self.h_crop, self.w_org)
        label_map = F.resize(label_map, size=(self.h_net, self.w_net), interpolation=T.InterpolationMode.NEAREST)
        
        # Load the RGB image
        with open(img_name, 'rb') as f:
            image = (Image.open(f).convert('RGB'))
        # image preprocess with crop and resize
        
        image = F.crop(image, self.h_crop, 0, self.h_org - self.h_crop, self.w_org)
        image = F.resize(image, size=(self.h_net, self.w_net), interpolation=T.InterpolationMode.BILINEAR)

        v = random.random()
        if self.data_aug and v < 0.5:
            image, label_map = data_aug_rotate(image, label_map)
            image, label_map = data_aug_randomflip(image, label_map)
            # label_map = Image.fromarray(label_map)
        image = self.totensor(image).float()
        # image = torch.from_numpy(image).long()
        image = self.normalize(image)
        # print(type(image), image)
        # print(type(label_map), np.array(label_map))
        label_map = torch.from_numpy(np.array(label_map, dtype=np.int32)).contiguous().long()

        # if self.transform:
        #     image, label = self.transform((image, label))
        #     image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        #     label = torch.from_numpy(label).contiguous().long()
        # /home/liang/Datasets/OpenLane/img/validation/segment-16979882728032305374_2719_000_2739_000_with_camera_labels/151865422154188200.jpg
        img_name = ops.join(*img_name.split('/')[6:])
        # print(img_name)
        return image, label_map, img_name, idx

    def init_dataset_3D(self, dataset_base_dir, txt_file_path):
        """
        :param dataset_info_file:
        :return: image paths, labels in normalized net input coordinates

        data processing:
        ground truth labels map are scaled wrt network input sizes
        """

        # load image path, and lane pts
        label_image_path = []
        gt_laneline_pts_all = []
        gt_class_label_all = []
        # gt_lane_visibility = []

        assert ops.exists(txt_file_path), '{:s} not exist'.format(txt_file_path)

        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # info_dict = json.loads(line)
                anno_path = ops.join(dataset_base_dir + '/lane3d_1000_v1.2/lane3d_1000', line[:-4] + 'json')  # remove sufix jpg and add lines.txt
                # anno_path = anno_path.replace('img', 'lane3d_1000_v1.2/lane3d_1000')
                with open(anno_path, 'r') as anno_file:
                    # data = [list(map(float, line.split())) for line in anno_file.readlines()]
                    info_dict = json.load(anno_file)

                image_path = ops.join(dataset_base_dir + '/img', info_dict['file_path'])
                assert ops.exists(image_path), '{:s} not exist'.format(image_path)
                gt_class = []
                gt_lane_pts = []
                for i in range(len(info_dict['lane_lines'])):
                    l = [(int(x), int(y)) for x, y in zip(info_dict['lane_lines'][i]['uv'][0], info_dict['lane_lines'][i]['uv'][1]) if x >= 0]
                    if (len(l)>3):
                    # gt_lane_pts.append(list(set(l)))
                        gt_lane_pts.append(l)
                        # gt_lane_visibility.append(info_dict['lane_lines'][i]['visibility'])
                        gt_class.append(info_dict['lane_lines'][i]['track_id'])
                # gt_lane_visibility = info_dict['laneLines_visibility']
                if len(gt_lane_pts) == 0:
                    continue

                label_image_path.append(image_path)
                gt_class_label_all.append(gt_class)

                # gt_lane_pts = [gt_lane_pts[ind] for ind in sort_id]
                gt_laneline_pts_all.append(gt_lane_pts)


        label_image_path = np.array(label_image_path)

        return label_image_path, gt_laneline_pts_all, gt_class_label_all



def data_aug_rotate(img, label):
    degree=(-10, 10)
    w = img.width
    h = img.height
    img = np.array(img)
    label = np.array(label)
    degree = random.uniform(degree[0], degree[1])
    center = (w / 2, h / 2)
    map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
    img = cv2.warpAffine(img, map_matrix, (w, h), flags=cv2.INTER_LINEAR)
    label = cv2.warpAffine(label, map_matrix, (w, h), flags=cv2.INTER_NEAREST)
    return img, label

def data_aug_randomflip(img, label):
    return Image.fromarray((np.fliplr(img))), Image.fromarray((np.fliplr(label)))


