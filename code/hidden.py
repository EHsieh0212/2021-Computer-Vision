import os
import numpy as np
import cv2
from tqdm import tqdm
from utils import AverageMeter
from testModel import my_awesome_algorithm



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
        image_folder = os.path.join(dataset_path, subject)
        for action_number in range(len([name for name in os.listdir(image_folder) if name !='.DS_Store'])):
            # if subject == 'KL':
            #     image_folder = os.path.join(dataset_path, subject, f'{action_number :04d}')
            # elif subject == 'HM':
            #     image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            image_folder = os.path.join(dataset_path, subject, f'{action_number + 1:02d}')
            try:
                os.mkdir('../hidden_result/'+subject+'/'+f'{action_number + 1:02d}')
            except:
                pass
            nr_image = len([name for name in os.listdir(image_folder) if name.endswith('.jpg')])
            #nr_image = len([name for name in os.listdir(image_folder)])
                #iou_meter_sequence.reset()
                #label_name = os.path.join(image_folder, '0.png')
                # if not os.path.exists(label_name):
                #     print(f'Labels are not available for {image_folder}')
                #     continue
            for idx in tqdm(range(nr_image), desc=f'[{sequence_idx:03d}] {image_folder}'):
                    image_name = os.path.join(image_folder, f'{idx}.jpg')

                    #label_name = os.path.join(image_folder, f'{idx}.png')
                    image = cv2.imread(image_name)
                    #label = cv2.imread(label_name)
                    # b,g,r = cv2.split(label)
                    # if not cv2.countNonZero(b) and not cv2.countNonZero(g) and not cv2.countNonZero(r):
                    #     l = 0
                    # else:
                    #     l = 1
                    # TODO: Modify the code below to run your method or load your results from disk
                    output, conf = my_awesome_algorithm(image)
                    path = r'../hidden_result/' + subject +'/'+f'{action_number + 1:02d}'
                    name = f'{idx}.png'
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
            sequence_idx+=1
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
        os.mkdir('../hidden_result')
    except:
        pass

    dataset_path = r'../dataset/hidden'
    subjects = []
    l = [name for name in os.listdir(dataset_path) if name !='.DS_Store']
    for i in l:
        subjects.append(i)
        try:
            os.mkdir('../hidden_result/'+i)
        except:
            pass
    subjects.sort()
    output, conf = benchmark(dataset_path, subjects)