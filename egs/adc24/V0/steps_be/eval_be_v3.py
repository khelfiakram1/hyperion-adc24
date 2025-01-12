#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import os
import logging
from jsonargparse import ArgumentParser, ActionYesNo, namespace_to_dict
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hyperion.hyp_defs import config_logger
from hyperion.utils import SegmentSet
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.helpers import VectorClassReader as VCR
from hyperion.np.transforms import TransformList
from hyperion.np.classifiers import LinearSVMC as SVM
from hyperion.np.metrics import compute_accuracy, compute_confusion_matrix, print_confusion_matrix
from hyperion.np.clustering import KMeans as KM

def compute_metrics(y_true, y_pred, labels):
    """
    Computes and logs metrics for the classifier, including accuracy and confusion matrices.
    """
    acc = compute_accuracy(y_true, y_pred)
    logging.info("test acc: %.2f %%", acc * 100)
    logging.info("non-normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=False)
    #C.plot_confusion_matrix() 
    print_confusion_matrix(C, labels)
    logging.info("normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=True)
    print_confusion_matrix(C * 100, labels, fmt=".2f")



dialect_groups = {
    "0": ["ara-arq", "ara-acm", "ara-arz", "ara-ayl", "ara-ksa", "ara-kuw",
    "ara-leb","ara-mau","ara-oma","ara-pal","ara-qat","ara-syr",
    "ara-uae",],
    
}

all_dialects = [
    'ara-acm', 'ara-arq', 'ara-arz', 'ara-ayl', 'ara-jor', 'ara-ksa', 'ara-kuw', 
    'ara-leb', 'ara-mau', 'ara-mor', 'ara-oma', 'ara-pal', 'ara-qat', 'ara-sud', 
    'ara-syr', 'ara-uae', 'ara-yem'
]

def evaluate_two_stage(
    v_file,
    trial_list,
    class_name,
    has_labels,
    svm,
    model_dir,
    score_file,
    verbose,
):
    config_logger(verbose)
    model_dir = Path(model_dir)
    output_dir = Path(score_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("loading data")
    segs = SegmentSet.load(trial_list)
    reader = DRF.create(v_file)
    x = reader.read(segs["id"], squeeze=True)
    x_bb=x
    del reader
    logging.info("loaded %d samples", x.shape[0])

   
    group_trans_file = model_dir / "transforms.h5"
    if group_trans_file.is_file():
        logging.info("loading group transform file %s", group_trans_file)
        group_trans = TransformList.load(group_trans_file)
        logging.info("applies group transform")
        x = group_trans(x)

   
    group_km_file = model_dir / "cluster.h5"
    logging.info("loading Group clustering file %s", group_km_file)
    group_km_model = KM.load_model(group_km_file)
    # logging.info("Group SVM args=%s", str(svm))
    logging.info("evals Group clustering")
    group_index,err = group_km_model.predict(x)

   
    #group_pred = np.argmax(group_scores, axis=-1)
    #predicted_groups = [group_km_model.labels[i] for i in group_pred]
    #logging.info(f"{len(predicted_groups)}")
    dialect_scores = []
    dialect_labels = []
    y_pred_dialects = []
    all_dialects = [
        'ara-acm', 'ara-arq', 'ara-arz', 'ara-ayl', 'ara-jor', 'ara-ksa', 'ara-kuw', 
        'ara-leb', 'ara-mau', 'ara-mor', 'ara-oma', 'ara-pal', 'ara-qat', 'ara-sud', 
        'ara-syr', 'ara-uae', 'ara-yem'
    ]
    for i, group in enumerate(group_index):
        #group_name = group_km_model.labels[group_pred[i]]
        #single_dialect_groups = {group: dialects for group, dialects in dialect_groups.items() if len(dialects) == 1}
        #binary_dialect_groups = {group: dialects for group, dialects in dialect_groups.items() if len(dialects) == 2}

        if group == 1 :
            y_pred_dialects.append("ara-jor")
            continue
        elif group == 2 :
            y_pred_dialects.append("ara-yem")
            continue
        elif group == 3 :
            y_pred_dialects.append("ara-mor")
            continue
        elif group == 4 :
            y_pred_dialects.append("ara-sud")
            continue


       
        group_model_path = model_dir / "cluster_0"
        
     
        dialect_trans_file = group_model_path / "transforms.h5"
        if dialect_trans_file.is_file():
            logging.info("loading dialect transform file %s", dialect_trans_file)
            dialect_trans = TransformList.load(dialect_trans_file)
            logging.info("applies dialect transform")
            
            x_i = x_bb[i:i+1, :] 
      
        
            x_i = dialect_trans(x_i)
        else:
            x_i = x_bb[i:i+1, :]  
        
        
        dialect_svm_file = group_model_path / "groups_model_svm.h5"
        logging.info("loading Dialect SVM file %s", dialect_svm_file)
        dialect_svm_model = SVM.load(dialect_svm_file)
        logging.info("Dialect SVM args=%s", str(svm))
        
        #logging.info("evals Dialect SVM for group %s", group_name)
        
 
        dialect_score = dialect_svm_model(x_i, **svm)


        logging.info(f"{np.argmax(dialect_score)}")
        dialect_labels.append(dialect_svm_model.labels)
        logging.info(f"{np.argmax(dialect_score, axis=-1)}")
        
        dialect_indices_to_labels = {index: label for index, label in enumerate(dialect_groups["0"])}
        
        pred_index = int(np.argmax(dialect_score, axis=-1)) 
        logging.info(f"{pred_index} is the predicted index")  
        predicted_dialect_label = dialect_indices_to_labels[pred_index]
        logging.info(predicted_dialect_label)
        y_pred_dialects.append(predicted_dialect_label)
       # logging.info(f"dialect score for original {class_ids[i]} is {dialect_score}")
    #logging.info(f" {np.array(dialect_scores).shape}")
    if has_labels:
        class_ids = segs[class_name].values
        # y_true_groups = np.asarray([group_km_model.labels.index(l) for l in class_ids])
        # y_pred_groups = group_pred
        
        # compute_metrics(y_true_groups, y_pred_groups, group_km_model.labels)

        y_true_dialects = []
       
        for i in range(len(class_ids)):
            true_dialect = class_ids[i]
            # logging.info(f"{true_dialect}")
            # if true_dialect in dialect_labels[i]:
            y_true_dialects.append(true_dialect)
            # y_pred_dialects.append(np.argmax(dialect_scores[i], axis=-1))
            # else:
            #     logging.warning(f"True dialect {true_dialect} is not in the list of predicted group {predicted_groups[i]} labels.")
            #     continue
        logging.info(f"{y_pred_dialects}")
        compute_metrics(y_true_dialects, y_pred_dialects,all_dialects)


    # Save scores
    # logging.info("Saving scores to %s", score_file)
    # score_table = {"segmentid": segs["id"]}
    
    # for i, group_label in enumerate(group_km_model.labels):
    #     score_table[f"group_{group_label}"] = group_scores[:, i]
    
    # for i, dialect_label_list in enumerate(dialect_labels):
    #     for j, dialect_label in enumerate(dialect_label_list):
    #         score_table[f"dialect_{dialect_label}"] = dialect_scores[i][:, j]

    # score_table = pd.DataFrame(score_table)
    # score_table.to_csv(score_file, sep="\t", index=False)
    


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Evaluate Two-Stage SVM Classifier",
    )

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--trial-list", required=True)
    SVM.add_eval_args(parser, prefix="svm")
    parser.add_argument("--class-name", default="class_id")
    parser.add_argument("--has-labels", default=True, action=ActionYesNo)
    parser.add_argument("--model-dir", required=True, help="Base directory containing the group classifier and all group models")
    parser.add_argument("--score-file", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    evaluate_two_stage(**namespace_to_dict(args))
