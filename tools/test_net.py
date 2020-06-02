# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from charnet.modeling.model import CharNet
import cv2, os
import numpy as np
import argparse
from charnet.config import cfg
import matplotlib.pyplot as plt


def save_word_recognition(word_instances, image_id, save_root, separator=chr(31)):
    with open('{}/{}.txt'.format(save_root, image_id), 'wt') as fw:
        for word_ins in word_instances:
            if len(word_ins.text) > 0:
                fw.write(separator.join([str(_) for _ in word_ins.word_bbox.astype(np.int32).flat]))
                fw.write(separator)
                fw.write(word_ins.text)
                fw.write(separator)
                char_texts = word_ins.char_texts
                char_bboxes = word_ins.char_bboxes
                for index, char_text in enumerate(char_texts):
                    char_bbox = char_bboxes[index]
                    fw.write(separator.join([str(_) for _ in char_bbox.astype(np.int32).flat]))
                    fw.write(separator)
                    fw.write(char_text)
                    if index != (char_texts.__len__() - 1):
                        fw.write(separator)
                fw.write('\n')

def save_char_recognition(char_instances, image_id, save_root, separator=chr(31)):
    with open('{}/{}.txt'.format(save_root, image_id), 'a') as fw:
        for char_ins in char_instances:
            if len(char_ins.text) > 0:
                fw.write(separator.join([str(_) for _ in char_ins.char_bbox.astype(np.int32).flat]))
                fw.write(separator)
                fw.write(char_ins.text)
                fw.write('\n')

def resize(im, size):
    h, w, _ = im.shape
    scale = max(h, w) / float(size)
    image_resize_height = int(round(h / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    image_resize_width = int(round(w / scale / cfg.SIZE_DIVISIBILITY) * cfg.SIZE_DIVISIBILITY)
    scale_h = float(h) / image_resize_height
    scale_w = float(w) / image_resize_width
    im = cv2.resize(im, (image_resize_width, image_resize_height), interpolation=cv2.INTER_LINEAR)
    return im, scale_w, scale_h, w, h


def vis(img, word_instances):
    img_word_ins = img.copy()
    # print(word_instances.__len__())
    for index, word_ins in enumerate(word_instances):
        word_bbox = word_ins.word_bbox
        cv2.polylines(img_word_ins, [word_bbox[:8].reshape((-1, 2)).astype(np.int32)],
                      True, (0, 0, 255), 2)
        cv2.putText(
            img_word_ins,
            '{}'.format(word_ins.text),
            (word_bbox[0], word_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

        for index, char_bbox in enumerate(word_ins.char_bboxes):
            char_text = word_ins.char_texts[index]
            cv2.polylines(img_word_ins, [char_bbox[:8].reshape((-1, 2)).astype(np.int32)], True, (0, 0, 255), 1)
            cv2.putText(img_word_ins, '{}'.format(char_text),
                (char_bbox[0], char_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
    return img_word_ins


def vis_char(img, char_instances):
    new_img = img.copy()
    for char_ins in char_instances:
        char_bbox = char_ins.char_bbox
        char_text  = char_ins.text
        cv2.polylines(new_img, [char_bbox[:8].reshape((-1, 2)).astype(np.int32)], True, (0, 0, 255), 1)
        cv2.putText(new_img, '{}'.format(char_text),
            (char_bbox[0], char_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    return new_img


def fit_char(char_bbox, im_original):
    array = char_bbox[:8].reshape((-1, 2)).astype(np.int32)
    pts = np.array(array, np.int32)
    [x_max, y_max] = np.amax(pts, axis=0)
    [x_min, y_min] = np.amin(pts, axis=0)
    w = x_max - x_min
    h = y_max - y_min

    half_x_min = int(x_min + h/4)
    half_y_min = int(y_min + h/4)
    half_h = int(h/2)
    half_w = int(w/2)

    is_correct_extend = [True, True, True, True]
    is_correct = False
    i=0
    while(not is_correct):
        is_correct_extend = [True, True, True, True]
        if((half_y_min > y_min) and ((half_y_min + half_h) < (y_min + h))):
            for x in range(half_x_min, half_x_min + half_w):
                # top
                if(all(i <=  200 for i in im_original[half_y_min, x])):
                    is_correct_extend[0] = False
                # btm
                if(all(i <= 200 for i in im_original[half_y_min + half_h, x])):
                    is_correct_extend[1] = False

            if(is_correct_extend[0] == False):
                half_y_min-=1
                half_h+=1

            if(is_correct_extend[1] == False):
                half_h+=1
        else:
            is_correct_extend[0] = True
            is_correct_extend[1] = True
            half_y_min = y_min
            half_h = h

        if((half_x_min > x_min) and ((half_x_min + half_w) < (x_min + w))):
            for y in range(half_y_min, half_y_min + half_h):
                # left
                if(all(i <=200 for i in im_original[y, half_x_min])):
                    is_correct_extend[2] = False
                # right
                if(all(i <= 200 for i in im_original[y, half_x_min + half_w])):
                    is_correct_extend[3] = False

            if(is_correct_extend[2] == False):
                half_x_min-=1
                half_w+=1

            if(is_correct_extend[3] == False):
                half_w+=1
        else:
            is_correct_extend[2] = True
            is_correct_extend[3] = True
            half_x_min = x_min
            half_w = w
        if(all(i ==  True for i in is_correct_extend)):
            is_correct = True

    half_x_max = half_x_min + half_w
    half_y_max = half_y_min + half_h
    return [half_x_min, half_y_min, half_x_max, half_y_min, half_x_max, half_y_max, half_x_min, half_y_max]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test")

    # parser.add_argument("config_file", help="path to config file", type=str)
    # parser.add_argument("image_dir", type=str)
    # parser.add_argument("results_dir", type=str)

    args = parser.parse_args()
    # args = {}
    args.config_file= "configs/icdar2015_hourglass88.yaml"
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    print(cfg)

    charnet = CharNet()
    charnet.load_state_dict(torch.load(cfg.WEIGHT))
    charnet.eval()
    charnet.cuda()
    args.image_dir = "/home/ngocnkd/project/research-charnet/datasets/icdar_sroie/icdar_sroie_train"
    args.results_dir = "/home/ngocnkd/project/research-charnet/outputs/icdar_sroie"
    # im_name = "X00016469669.jpg"
    
    # im_file = os.path.join(args.image_dir, im_name)
    # im_original = cv2.imread(im_file)
    # im, scale_w, scale_h, original_w, original_h = resize(im_original, size=cfg.INPUT_SIZE)

    for im_name in sorted(os.listdir(args.image_dir)):
        print("Processing {}...".format(im_name))
        im_file = os.path.join(args.image_dir, im_name)
        im_original = cv2.imread(im_file)
        im, scale_w, scale_h, original_w, original_h = resize(im_original, size=cfg.INPUT_SIZE)
        with torch.no_grad():
            char_bboxes, char_scores, word_instances, char_instances= charnet(im, scale_w, scale_h, original_w, original_h)

            # word_ins = word_instances[0]
            # word_bbox = word_ins.word_bbox
            # cv2.polylines(im_original, [word_bbox[:8].reshape((-1, 2)).astype(np.int32)],
            #             True, (0, 0, 255), 2)
            # cv2.putText(
            #     im_original,
            #     '{}'.format(word_ins.text),
            #     (word_bbox[0], word_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            # )
            # for index, char_text in enumerate(word_ins.char_texts):
            #     char_bbox = word_ins.char_bboxes[index]
            #     # char_bbox[]
            #     x_max = max(char_bbox[0::2])
            #     x_min = min(char_bbox[0::2])
            #     y_max = max(char_bbox[1::2])
            #     y_min = min(char_bbox[1::2])
            #     cv2.rectangle(im_original, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            #     cv2.putText(
            #     im_original,
            #     '{}'.format(char_text),
            #     (char_bbox[0], char_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            #     )
            # cv2.imwrite("outputs/test/"+ im_name, im_original)
            # print(np.amax(char_bboxes, axis=0))
            # print(np.amin(char_bboxes, axis=0))
            for i, word_ins in enumerate(word_instances):
                for j, char_bbox in enumerate(word_ins.char_bboxes):
                    word_instances[i].char_bboxes[j] = fit_char(char_bbox, im_original)

            # cv2.rectangle(new_img, (half_x_min, half_y_min), (half_x_min + half_w, half_y_min + half_h), (0, 255, 0), 1)
            new_img = vis(im_original, word_instances)
            new_img = vis_char(new_img, char_instances)
            cv2.imwrite("outputs/test/"+ im_name, new_img)

            save_word_recognition(
                word_instances, os.path.splitext(im_name)[0],
                args.results_dir, cfg.RESULTS_SEPARATOR
            )
            save_char_recognition(char_instances, os.path.splitext(im_name)[0],
                args.results_dir, cfg.RESULTS_SEPARATOR
            )
