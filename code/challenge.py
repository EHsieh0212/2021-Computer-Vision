import os
import numpy as np
import cv2
from tqdm import tqdm
from utils import AverageMeter
from testModel import my_awesome_algorithm
import shutil



def benchmark(dataset_path: str, subjects: list):
    """Compute the weighted IoU and average true negative rate
    Args:
        dataset_path: the dataset path
        subjects: a list of subject names

    Returns: benchmark score

    """
    output_conf = []
    sequence_idx = 0
    for subject in subjects:
        # if subject == 'KL':
        #     num = 300
        # elif subject == 'HM':
        #     num = 12
        #for action_number in range(num):
            # if subject == 'KL':
            #     image_folder = os.path.join(dataset_path, subject, f'{action_number :04d}')
            # elif subject == 'HM':
            #     image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
        image_folder = os.path.join(dataset_path, subject)
            #nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
        nr_image = len([name for name in os.listdir(image_folder)])
            #iou_meter_sequence.reset()
            #label_name = os.path.join(image_folder, '0.png')
            # if not os.path.exists(label_name):
            #     print(f'Labels are not available for {image_folder}')
            #     continue
        for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                if subject == 'KL':
                    name = f'{idx:04d}.jpg'
                    image_name = os.path.join(image_folder, name)
                elif subject == 'HM':
                    if idx+1 >= 11:
                        name = f'{idx+1:02d}.png'
                        image_name = os.path.join(image_folder, name)
                    else:
                        name = f'{idx+1:02d}.jpg'
                        image_name = os.path.join(image_folder, name)
                #label_name = os.path.join(image_folder, f'{idx}.png')
                image = cv2.imread(image_name)
                #label = cv2.imread(label_name)
                # b,g,r = cv2.split(label)
                # if not cv2.countNonZero(b) and not cv2.countNonZero(g) and not cv2.countNonZero(r):
                #     l = 0
                # else:
                #     l = 1
                # f=open('labels.txt','a')
                # f.write(image_name+' '+str(l)+'\n')
                # TODO: Modify the code below to run your method or load your results from disk
                output, conf = my_awesome_algorithm(image)
                path = r'../challenge_result/' + subject
                cv2.imwrite(os.path.join(path , name), output*255)
                f=open(os.path.join(path , 'conf.txt'),'a')
                f.write(str(conf) + '\n')
                # output = label
                # conf = 1.0
                # if np.sum(label.flatten()) > 0:
                #     label_validity.append(1.0)
                #     iou = mask_iou(output, label)
                #     iou_meter.update(conf * iou)
                #     iou_meter_sequence.update(conf * iou)
                # else:  # empty ground truth label
                #     label_validity.append(0.0)
                # output_conf.append(conf)
            # print(f'[{sequence_idx:03d}] Weighted IoU: {iou_meter_sequence.avg()}')
    # tn_rates = true_negative_curve(np.array(output_conf), np.array(label_validity))
    # wiou = iou_meter.avg()
    # atnr = np.mean(tn_rates)
    # score = 0.7 * wiou + 0.3 * atnr
    # print(f'\n\nOverall weighted IoU: {wiou:.4f}')
    # print(f'Average true negative rate: {atnr:.4f}')
    # print(f'Benchmark score: {score:.4f}')

    return output, conf


if __name__ == '__main__':
    try:
        shutil.rmtree('../challenge_result')
        print("remove '../challenge_result'")
    except Exception as e:
        print(e)

    try:
        os.mkdir('../challenge_result')
    except:
        print("'../challenge_result' already exist")
    
    try:
        os.mkdir('../challenge_result/KL')
    except:
        print("'../challenge_result/KL' already exist")
    
    try:
        os.mkdir('../challenge_result/HM')
    except:
        print("'../challenge_result/HM' already exist")

    dataset_path = r'../dataset/CV22S_Ganzin_challenge'
    subjects = ['KL','HM']
    output, conf = benchmark(dataset_path, subjects) 