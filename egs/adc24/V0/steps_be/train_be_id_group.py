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

from hyperion.hyp_defs import config_logger
from hyperion.utils import SegmentSet
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.helpers import VectorClassReader as VCR
from hyperion.np.transforms import TransformList, PCA, LNorm
from hyperion.np.classifiers import LinearSVMC as SVM
from hyperion.np.metrics import compute_accuracy, compute_confusion_matrix, print_confusion_matrix
from hyperion.np.classifiers import BinaryLogisticRegression as LR

def compute_metrics(y_true, y_pred, labels, group_name):
   
    acc = compute_accuracy(y_true, y_pred)
    logging.info(f"Group {group_name}: Training accuracy: %.2f %%", acc * 100)
    
    logging.info(f"Group {group_name}: Non-normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=False)
    print_confusion_matrix(C, labels)
    
    logging.info(f"Group {group_name}: Normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=True)
    print_confusion_matrix(C * 100, labels)


def train_be(
    v_file,
    train_list,
    class_name,
    do_lnorm,
    whiten,
    pca,
    svm,
    output_dir,
    verbose,
):
  
    config_logger(verbose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    
   
   
  
    train_segs = SegmentSet.load(train_list)
    dialects_groups = {
    "Maghrebi1": ["ara-arq", "ara-mor", "ara-mau","ara-ayl"],
    "EgyptianSudanese": ["ara-arz", "ara-sud","ara-pal", "ara-jor"],
    "Levantine": [ "ara-leb", "ara-syr","ara-yem"],
    "Gulf": ["ara-ksa", "ara-kuw", "ara-oma", "ara-qat", "ara-uae"],
    "Mesopotamian": ["ara-acm"]
    }
  
    single_dialect_groups = {group: dialects for group, dialects in dialects_groups.items() if len(dialects) == 1}
    for group_name, dialects in dialects_groups.items():
        if group_name in single_dialect_groups :
            continue
        
        logging.info(f"Training classifier for group: {group_name}")

      
        filtered_train_segs = train_segs.df[train_segs.df[class_name].isin(dialects)]
        filtered_train_segs = SegmentSet(df=filtered_train_segs)

     
        group_output_dir = output_dir / group_name
        group_output_dir.mkdir(parents=True, exist_ok=True)


        train_reader = DRF.create(v_file)
        x_trn = train_reader.read(filtered_train_segs["id"], squeeze=True)
        del train_reader
        class_ids = filtered_train_segs[class_name]
        labels, y_true = np.unique(class_ids, return_inverse=True)
        logging.info("loaded %d samples for group %s", x_trn.shape[0], group_name)

        # Check PCA parameters and create PCA object if needed
        # if pca.get("pca_var_r") is not None and pca["pca_var_r"] < 1.0 or pca.get("pca_dim") is not None:
        #     logging.info("PCA args=%s", str(pca))
        #     logging.info("training PCA for group %s", group_name)
        #     pca_transform = PCA(**pca)
        #     pca_transform.fit(x_trn)
        #     logging.info("PCA dimension: %d", pca_transform.pca_dim)
        #     logging.info("apply PCA for group %s", group_name)
        #     x_trn = pca_transform(x_trn)
        # else:
        pca_transform = None

     
        if do_lnorm:
            lnorm = LNorm()
            if whiten:
                logging.info("training whitening for group %s", group_name)
                lnorm.fit(x_trn)
            logging.info("apply lnorm for group %s", group_name)
            x_trn = lnorm(x_trn)
        else:
            lnorm = None

    
        logging.info("SVM args=%s", str(svm))
        model = SVM(labels=labels, **svm)
        model.fit(x_trn, y_true)
        logging.info("trained SVM for group %s", group_name)


        scores = model(x_trn)
        y_pred = np.argmax(scores, axis=-1)
        compute_metrics(y_true, y_pred, labels, group_name)


        logging.info(f"Saving transforms and SVM for group {group_name}")
        transforms = []
        if pca_transform is not None:
            transforms.append(pca)
        if lnorm is not None:
            transforms.append(lnorm)

        if transforms:
            transforms = TransformList(transforms)
            transforms.save(group_output_dir / f"transforms_{group_name}.h5")

        model.save(group_output_dir / f"model_svm_{group_name}.h5")


if __name__ == "__main__":

    parser = ArgumentParser(
        description="Train linear SVM Classifier",
    )

    parser.add_argument("--v-file", required=True)
    parser.add_argument("--train-list", required=True)
    PCA.add_class_args(parser, prefix="pca")
    SVM.add_class_args(parser, prefix="svm")
    parser.add_argument("--class-name", default="class_id")
    parser.add_argument("--do-lnorm", default=True, action=ActionYesNo)
    parser.add_argument("--whiten", default=True, action=ActionYesNo)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    

    train_be(**namespace_to_dict(args))
