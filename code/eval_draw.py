from curses import savetty
import os
import numpy as np
import cv2
from tqdm import tqdm
from utils import AverageMeter
from testModel import my_awesome_algorithm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import logging
from os import walk
import argparse
import shutil

FORMAT = '%(asctime)s  %(levelname)s  %(message)s'
logging.basicConfig(
    format=FORMAT, filename="./log_eval.txt", level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--only_s5', help='feature', type=bool, default=False)
parser.add_argument('--save_result', help='classifier', type=bool, default=False)
args = parser.parse_args()
USE_S5 = args.only_s5
SAVE_RESULT = args.save_result

def true_negative_curve(confs: np.ndarray, labels: np.ndarray, nr_thresholds: int = 1000):
    """Compute true negative rates
    Args:
        confs: the algorithm outputs
        labels: the ground truth labels
        nr_thresholds: number of splits for sliding thresholds

    Returns:

    """
    thresholds = np.linspace(0, 1, nr_thresholds)
    tn_rates = []
    for th in thresholds:
        # thresholding
        predict_negatives = (confs < th).astype(int)
        # true negative
        tn = np.sum((predict_negatives * (1 - labels) > 0).astype(int))
        tn_rates.append(tn / np.sum(1 - labels))
    return np.array(tn_rates)


def mask_iou(mask1: np.ndarray, mask2: np.ndarray):
    """Calculate the IoU score between two segmentation masks
    Args:
        mask1: 1st segmentation mask
        mask2: 2nd segmentation mask
    """
    if len(mask1.shape) == 3:
        mask1 = mask1.sum(axis=-1)
    if len(mask2.shape) == 3:
        mask2 = mask2.sum(axis=-1)
    area1 = cv2.countNonZero((mask1 > 0).astype(int))
    area2 = cv2.countNonZero((mask2 > 0).astype(int))
    if area1 == 0 or area2 == 0:
        return 0
    area_union = cv2.countNonZero(((mask1 + mask2) > 0).astype(int))
    area_inter = area1 + area2 - area_union
    return area_inter / area_union


def benchmark(subject):
    logging.info(f'start {subject}')
    """Compute the weighted IoU and average true negative rate
    Args:
        dataset_path: the dataset path
        subjects: a list of subject names

    Returns: benchmark score

    """

    dataset_path = "./../dataset/public"
    if USE_S5:
        try:
            shutil.rmtree(os.path.join(dataset_path, 'S5_solution').replace("\\", "/"))
            print("remove")
        except Exception as e:
            logging.warning(f"Exception of rm s5_solution folder: {e}")

        try:
            os.mkdir(os.path.join(dataset_path, 'S5_solution').replace("\\", "/"))
        except Exception as e:
            logging.warning(f"Exception of create s5_solution folder: {e}")
        

    iou_meter = AverageMeter()
    iou_meter_sequence = AverageMeter()
    label_validity = []
    output_conf = []
    sequence_idx = 0
    s_path = os.path.join(dataset_path, subject)
    c_folders = next(walk(s_path), (None, None, []))[1]

    for c in c_folders:
        logging.info(f"start {c} folder")
        image_folder = os.path.join(s_path, c)
        sequence_idx += 1
        nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
        iou_meter_sequence.reset()
        label_name = os.path.join(image_folder, '0.png')
        if SAVE_RESULT:
            try:
                 os.mkdir(os.path.join(*[dataset_path, 'S5_solution', c]))
            except Exception as e:
                logging.warning(f"Exception of create s5_solution/{c} folder: {e}")
        
        if not USE_S5 and not os.path.exists(label_name):
            logging.warning(f'Labels are not available for {image_folder}')
            continue
        for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
            image_name = os.path.join(image_folder, f'{idx}.jpg')
            image = cv2.imread(image_name)
            if not USE_S5:
                label_name = os.path.join(image_folder, f'{idx}.png')
                label = cv2.imread(label_name)
            # TODO: Modify the code below to run your method or load your results from disk
            output, conf = my_awesome_algorithm(image)
            if SAVE_RESULT and USE_S5:
                label_name = os.path.join(image_folder, f'{idx}.png')
                result_name = label_name.replace("S5","S5_solution")
                cv2.imwrite(result_name, output*255)
                conf_path = os.path.join(image_folder, "conf.txt")
                conf_path = conf_path.replace("S5","S5_solution")
                with open(conf_path, 'a+') as t:
                    t.write(str(conf)+"\n")
                logging.info(f"save data to {result_name}")
                logging.info(f"save conf {conf_path}")
                continue
            # output = label
            # conf = 1.0
            if np.sum(label.flatten()) > 0:
                label_validity.append(1.0)
                iou = mask_iou(output, label)
                iou_meter.update(conf * iou)
                iou_meter_sequence.update(conf * iou)
            else:  # empty ground truth label
                label_validity.append(0.0)
            output_conf.append(conf)
        # print(f'[{sequence_idx:03d}] Weighted IoU: {iou_meter_sequence.avg()}')
    if USE_S5:
        return None, None

    tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
    
    wiou = iou_meter.avg()
    atnr = np.mean(tn_rates)
    score = 0.7 * wiou + 0.3 * atnr
    
    logging.info(f'{subject}-weighted IoU: {wiou:.4f}')
    logging.info(f'{subject}-true negative rate: {atnr:.4f}')
    logging.info(f'{subject}-Benchmark score: {score:.4f}')

    return iou_meter, (np.array(output_conf), np.array(label_validity))


def main(inputs):
    
    benchmark(inputs[0])


    
if __name__ == '__main__':
    if USE_S5 and SAVE_RESULT:
        subjects = ['S5']
    else:
        subjects = ['S1', 'S2', 'S3', 'S4']
    
    main(subjects)
