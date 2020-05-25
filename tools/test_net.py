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
    for word_ins in word_instances:
        word_bbox = word_ins.word_bbox
        cv2.polylines(img_word_ins, [word_bbox[:8].reshape((-1, 2)).astype(np.int32)],
                      True, (0, 255, 0), 2)
        cv2.putText(
            img_word_ins,
            '{}'.format(word_ins.text),
            (word_bbox[0], word_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    return img_word_ins

def vis_char(img, char_bboxes, char_text_array):
    new_img = img.copy()
    for index, char_bbox in enumerate(char_bboxes):
        cv2.polylines(new_img, [char_bbox[:8].reshape((-1, 2)).astype(np.int32)],
                    True, (0, 0, 255), 1)
        cv2.putText(new_img,
            '{}'.format(char_text_array[index]),
            (char_bbox[0], char_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )
    return new_img

def box_extend(img, old_loc):
    new_loc = []


    return new_loc


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
    im_name = "X00016469612.jpg"
    im_file = os.path.join(args.image_dir, im_name)
    im_original = cv2.imread(im_file)
    im, scale_w, scale_h, original_w, original_h = resize(im_original, size=cfg.INPUT_SIZE)


    with torch.no_grad():
        char_bboxes, char_scores, word_instances, char_text_array, char_score_array = charnet(im, scale_w, scale_h, original_w, original_h)
        # save_word_recognition(
        #     word_instances, os.path.splitext(im_name)[0],
        #     args.results_dir, cfg.RESULTS_SEPARATOR
        # )

        # new_img = vis(im_original, word_instances)
        # print(np.amax(char_bboxes, axis=0))
        # print(np.amin(char_bboxes, axis=0))
        for index, char_bbox in enumerate(char_bboxes):
            cv2.polylines(im_original, [char_bbox[:8].reshape((-1, 2)).astype(np.int32)],
                        True, (0, 0, 255), 1)
            array = char_bbox[:8].reshape((-1, 2)).astype(np.int32)
            # print(char_bbox[:8].reshape((-1, 2)).astype(np.int32))
            # cv2.putText(
            #     result,
            #     '{}'.format(char_text_array[index]),
            #     (char_bbox[0], char_bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            # )

            # array = [[73, 30], [88, 30], [73, 55], [88, 55]]
            pts = np.array(array, np.int32)
            # print(pts/2)
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
            # print("Before: ", half_x_min, half_y_min, half_x_min + half_w, half_y_min + half_h)
            while(not is_correct):
                # cv2.rectangle(im_original, (half_x_min, half_y_min), (half_x_min + half_w, half_y_min + half_h), (0, 255, 0), 1)
                # cv2.imwrite("outputs/demo1.jpg", im_original)
                # all(i ==  True for i in is_correct_extend)):
                # print("Count:", i)
                # print(is_correct_extend)

                is_correct_extend = [True, True, True, True]
                if((half_y_min > y_min) and ((half_y_min + half_h) < (y_min + h))):
                    for x in range(half_x_min, half_x_min + half_w):
                        # top
                        print(im_original[half_y_min, x])
                        print(im_original[half_y_min + half_h, x])

                        if(all(i <=  200 for i in im_original[half_y_min, x])):
                            is_correct_extend[0] = False
                            break
                        # btm
                        if(all(i <= 200 for i in im_original[half_y_min + half_h, x])):
                            is_correct_extend[1] = False
                            break

                    if(is_correct_extend[0] == False):
                        half_y_min-=1
                        half_h+=1
                        # cv2.rectangle(im_original, (half_x_min, half_y_min), (half_x_min + half_w, half_y_min + half_h), (0, 255, 0), 1)
                        continue

                    if(is_correct_extend[1] == False):
                        half_h+=1
                        # cv2.rectangle(im_original, (half_x_min, half_y_min), (half_x_min + half_w, half_y_min + half_h), (0, 255, 0), 1)
                        continue
                else:
                    is_correct_extend[0] = True
                    is_correct_extend[1] = True

                if((half_x_min > x_min) and ((half_x_min + half_w) < (x_min + w))):
                    for y in range(half_y_min, half_y_min + half_h):
                        # left
                        # print(im_original[y, half_x_min])
                        # print(im_original[y, half_x_min + half_w])
                        if(all(i <=200 for i in im_original[y, half_x_min])):
                            is_correct_extend[2] = False
                            break
                        # right
                        if(all(i <= 200 for i in im_original[y, half_x_min + half_w])):
                            is_correct_extend[3] = False
                            break

                    if(is_correct_extend[2] == False):
                        half_x_min-=1
                        half_w+=1
                        continue

                    if(is_correct_extend[3] == False):
                        half_w+=1
                        continue
                else:
                    is_correct_extend[2] = True
                    is_correct_extend[3] = True
                if(all(i ==  True for i in is_correct_extend)):
                    is_correct = True

            cv2.rectangle(im_original, (half_x_min, half_y_min), (half_x_min + half_w, half_y_min + half_h), (0, 255, 0), 1)
            cv2.imwrite("outputs/demo1.jpg", im_original)

        # cv2.rectangle(im_original, (x_min, y_min), (x_min + w, y_min + h), (0, 0, 255), 1)



        # print(word_instances)
        # print(char_bboxes, char_scores)

    # for im_name in sorted(os.listdir(args.image_dir)):
    #     print("Processing {}...".format(im_name))
    #     im_file = os.path.join(args.image_dir, im_name)
    #     im_original = cv2.imread(im_file)
    #     im, scale_w, scale_h, original_w, original_h = resize(im_original, size=cfg.INPUT_SIZE)
    #     with torch.no_grad():
    #         char_bboxes, char_scores, word_instances, char_text_array, char_score_array = charnet(im, scale_w, scale_h, original_w, original_h)
    #         # save_word_recognition(
    #         #     word_instances, os.path.splitext(im_name)[0],
    #         #     args.results_dir, cfg.RESULTS_SEPARATOR
    #         # )
    #         new_img = vis_char(im_original, char_bboxes, char_text_array)
    #         cv2.imwrite(os.path.join(args.results_dir, im_name), new_img)
