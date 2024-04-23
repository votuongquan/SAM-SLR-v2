from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import argparse
import pickle
import random


import numpy as np
from tqdm import tqdm

label = open('/kaggle/input/inference-for-slr-v2/27/test_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('/kaggle/working/SAM-SLR-v2/SL-GCN/work_dir/vsl_inference_joint_test/eval_results/best_acc.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('/kaggle/working/SAM-SLR-v2/SL-GCN/work_dir/vsl_inference_bone_test/eval_results/best_acc.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open('/kaggle/working/SAM-SLR-v2/SL-GCN/work_dir/vsl_inference_joint_motion_test/eval_results/best_acc.pkl', 'rb')
r3 = list(pickle.load(r3).items())
r4 = open('/kaggle/working/SAM-SLR-v2/SL-GCN/work_dir/vsl_inference_bone_motion_test/eval_results/best_acc.pkl', 'rb')
r4 = list(pickle.load(r4).items())

alpha = [1.0, 0.9, 0.5, 0.5]  # 51.50

right_num = total_num = right_num_5 = 0
names = []
preds = []
scores = []
mean = 0

with open('predictions.csv', 'w') as f:

    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        names.append(name)
        name1, r11 = r1[i]
        name2, r22 = r2[i]
        name3, r33 = r3[i]
        name4, r44 = r4[i]
        try:
            assert name == name1 == name2 == name3 == name4
        except AssertionError:
            print(
                f"Assertion failed for names: {name}, {name1}, {name2}, {name3}, {name4}")
            raise
        mean += r11.mean()
        score = (r11*alpha[0] + r22*alpha[1] + r33 *
                 alpha[2] + r44*alpha[3]) / np.array(alpha).sum()
        rank_5 = score.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(score)
        scores.append(score)
        preds.append(r)
        right_num += int(r == int(l))
        total_num += 1
        f.write('{}, {}\n'.format(name, r))
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(total_num)
    print('top1: ', acc)
    print('top5: ', acc5)

f.close()
print(mean/len(label[0]))

with open('./gcn_ensembled.pkl', 'wb') as f:
    score_dict = dict(zip(names, scores))
    pickle.dump(score_dict, f)


while True:
    print("Number of samples: ", total_num)
    # Let the user choose randomly an index to show predictions and ground truth
    index = int(
        input(f"\nEnter an index to show predictions and ground truth: "))
    # ensure the index is within the range of the total number of samples
    while index < 0 or index >= total_num:
        print("Index out of range. Please enter a valid index.")
        index = int(
            input("Enter an index to show predictions and ground truth: "))
    print("Prediction: ", preds[index])
    print("Ground Truth: ", label[1, index])

    # ask the user if they want to continue
    cont = input("Do you want to test another one? (yes/no): ")
    if cont.lower() != "yes":
        break

# Ask the user if they want to see the classification report
print("Do you want to see the classification report? (yes/no): ")
show_report = input()
if show_report.lower() == "yes":
    # make a classification report

    y_true = [int(l) for _, l in label.T]
    y_pred = preds
    print("Classification Report")
    print(classification_report(y_true, y_pred))

    # save classification report
    with open('./ensemble_classification_report.txt', 'w') as f:
        f.write(classification_report(y_true, y_pred))

print("Ensemble predictions saved in predictions.csv")
