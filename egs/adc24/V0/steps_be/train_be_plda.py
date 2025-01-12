#!/usr/bin/env python
"""
 Copyright 2018 Johns Hopkins University  (Author: Jesus Villalba)
 Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""

import sys
import os
import logging
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
    ActionParser,
    namespace_to_dict,
    ActionYesNo,
)
import time
from pathlib import Path

import numpy as np

from hyperion.hyp_defs import config_logger
from hyperion.utils import SegmentSet
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.helpers import VectorClassReader as VCR
from hyperion.np.transforms import TransformList, PCA, LNorm
from hyperion.np.classifiers import LinearSVMC as SVM
from hyperion.np.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    print_confusion_matrix,
)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC 

def compute_metrics(y_true, y_pred, labels):

    acc = compute_accuracy(y_true, y_pred)
    logging.info("training acc: %.2f %%", acc * 100)
    logging.info("non-normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=False)
    print_confusion_matrix(C, labels)
    logging.info("normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=True)
    print_confusion_matrix(C * 100, labels)


def map_dialect_to_groups(dialect, dialect_groups):
    
    groups = []
    for group_name, dialect_list in dialect_groups.items():
        if dialect in dialect_list:
            groups.append(group_name)
    return groups

def prepare_data_with_multi_group_labels(train_list, v_file, class_name, dialect_groups):
    """
    Prepares the training data, assigning each sample multiple group labels based on the dialect.
    """
    logging.info("Loading data...")
    train_segs = SegmentSet.load(train_list)
    train_reader = DRF.create(v_file)
    x_trn = train_reader.read(train_segs["id"], squeeze=True)
    del train_reader

    # Map each dialect to all its groups
    class_ids = train_segs[class_name]
    multi_group_labels = [map_dialect_to_groups(dialect, dialect_groups) for dialect in class_ids]
    
    # Create a binary matrix for group labels (multi-label format)
    unique_groups = list(dialect_groups.keys())
    y_true_groups = np.zeros((len(multi_group_labels), len(unique_groups)), dtype=int)
    
    for i, groups in enumerate(multi_group_labels):
        for group in groups:
            group_index = unique_groups.index(group)
            y_true_groups[i, group_index] = 1

    logging.info("Loaded %d samples", x_trn.shape[0])
    return x_trn, y_true_groups, unique_groups


def train_and_save_multi_label_group_classifier(v_file, train_list, class_name, do_lnorm, whiten, pca, svm, output_dir, verbose, dialect_groups):
    config_logger(verbose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data with multi-group labels
    x_trn, y_true_groups, unique_group_labels = prepare_data_with_multi_group_labels(train_list, v_file, class_name, dialect_groups)

    # Apply PCA if needed
    if pca:
        logging.info("Training PCA...")
        pca = PCA(**pca)
        pca.fit(x_trn)
        logging.info("PCA dimension: %d", pca.pca_dim)
        logging.info("Applying PCA...")
        x_trn = pca(x_trn)

    # Apply normalization if needed
    if do_lnorm:
        lnorm = LNorm()
        if whiten:
            logging.info("Training whitening normalization...")
            lnorm.fit(x_trn)
        logging.info("Applying normalization...")
        x_trn = lnorm(x_trn)
    
    # Train the group classifier using OneVsRest for multi-label classification
    logging.info("Training Multi-Label Group Classifier using OneVsRest SVM...")
    base_svm = SVC(**svm)  # Base SVM model for each group
    group_svm = OneVsRestClassifier(base_svm)
    group_svm.fit(x_trn, y_true_groups)
    logging.info("Trained Multi-Label Group SVM with OneVsRest")

    # Save the group classifier model
    model_path = output_dir / "group_model_svm.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(group_svm, f)
    logging.info(f"Saved Group Classifier model to {model_path}")

    # Compute and print metrics
    y_pred = group_svm.predict(x_trn)
    for i, group in enumerate(unique_group_labels):
        compute_metrics(y_true_groups[:, i], y_pred[:, i], [0, 1])
    
    return group_svm, pca, lnorm
def train_and_save_dialect_classifiers(v_file, train_list, class_name, group_svm, pca, lnorm, output_dir, verbose, dialect_groups):
    config_logger(verbose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_segs = SegmentSet.load(train_list)
    train_reader = DRF.create(v_file)
    x_trn = train_reader.read(train_segs["id"], squeeze=True)
    del train_reader

    # Apply PCA and normalization
    if pca:
        x_trn = pca(x_trn)
    if lnorm:
        x_trn = lnorm(x_trn)

    # Predict groups for each sample using the trained multi-label group SVM
    group_scores = group_svm(x_trn)
    group_predictions = (group_scores > 0.5).astype(int)

    group_dialect_classifiers = {}
    for group_idx, group_label in enumerate(group_svm.labels):
        # Get indices of samples predicted as this group
        group_indices = np.where(group_predictions[:, group_idx] == 1)[0]
        
        if len(group_indices) == 0:
            logging.info(f"No samples predicted for group {group_label}, skipping.")
            continue

        x_group = x_trn[group_indices]
        class_ids = train_segs[class_name][group_indices]
        dialect_labels, y_true_group = np.unique(class_ids, return_inverse=True)

        # Train the dialect classifier for this group
        logging.info(f"Training Dialect Classifier for group {group_label}")
        dialect_svm = SVM(labels=dialect_labels, **svm)
        dialect_svm.fit(x_group, y_true_group)
        logging.info(f"Trained Dialect SVM for group {group_label}")

        # Save the dialect classifier
        dialect_svm.save(output_dir / f"dialect_model_svm_{group_label}.h5")
        logging.info(f"Saved Dialect Classifier model for group {group_label} to {output_dir / f'dialect_model_svm_{group_label}.h5'}")

        group_dialect_classifiers[group_label] = dialect_svm

    return group_dialect_classifiers

def predict_dialect(x, group_svm, group_dialect_classifiers, pca=None, lnorm=None):
    # Apply PCA and normalization
    if pca:
        x = pca(x)
    if lnorm:
        x = lnorm(x)
    
    # Stage 1: Predict the group (multi-label)
    group_scores = group_svm(x)
    group_preds = (group_scores > 0.5).astype(int)
    predicted_groups = [group_svm.labels[i] for i in range(len(group_preds[0])) if group_preds[0, i] == 1]
    
    logging.info(f"Predicted groups: {predicted_groups}")
    
    # Stage 2: Predict the specific dialect within each predicted group
    predicted_dialects = {}
    for group in predicted_groups:
        dialect_svm = group_dialect_classifiers.get(group)
        if dialect_svm is None:
            logging.error(f"No dialect classifier found for group {group}.")
            continue
        
        dialect_scores = dialect_svm(x)
        dialect_pred = np.argmax(dialect_scores, axis=-1)
        predicted_dialect = dialect_svm.labels[dialect_pred[0]]
        predicted_dialects[group] = predicted_dialect

        logging.info(f"Predicted dialect for group {group}: {predicted_dialect}")

    return predicted_groups, predicted_dialects

if __name__ == "__main__":
    parser = ArgumentParser(description="Train Two-Stage SVM Classifiers")
    parser.add_argument("--v-file", required=True)
    parser.add_argument("--train-list", required=True)
    
    # Add class args for PCA and SVM
    PCA.add_class_args(parser, prefix="pca")
    SVM.add_class_args(parser, prefix="svm")
    
    parser.add_argument("--class-name", default="class_id")
    parser.add_argument("--do-lnorm", default=True, action=ActionYesNo)
    parser.add_argument("--whiten", default=True, action=ActionYesNo)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int)
    
    args = parser.parse_args()

    # Convert parsed arguments to a dictionary
    args_dict = vars(args)

    # Manually filter PCA and SVM arguments
    pca_args = {k[len('pca_'):]: v for k, v in args_dict.items() if k.startswith('pca_')}
    svm_args = {k[len('svm_'):]: v for k, v in args_dict.items() if k.startswith('svm_')}

    # Define dialect groups with multiple group memberships
    dialect_groups = {
        "Maghrebi1": ["ara-arq", "ara-mor", "ara-mau"],
        "Maghrebi2": ["ara-arq", "ara-ayl"],
        "EgyptianSudanese": ["ara-arz", "ara-ayl", "ara-sud"],
        "EPL": ["ara-arz", "ara-pal", "ara-jor", "ara-syr"],
        "Levantine": ["ara-jor", "ara-leb", "ara-pal", "ara-syr"],
        "Gulf": ["ara-ksa", "ara-kuw", "ara-oma", "ara-qat", "ara-uae", "ara-yem"],
        "Mesopotamian": ["ara-acm"]
    }

    # Stage 1: Train and Save the Multi-Label Group Classifier
    group_svm, pca_transform, lnorm_transform = train_and_save_multi_label_group_classifier(
        v_file=args.v_file,
        train_list=args.train_list,
        class_name=args.class_name,
        do_lnorm=args.do_lnorm,
        whiten=args.whiten,
        pca=pca_args,   # Correctly passed PCA arguments
        svm=svm_args,   # Correctly passed SVM arguments
        output_dir=args.output_dir,
        verbose=args.verbose,
        dialect_groups=dialect_groups
    )

    # Stage 2: Train and Save Dialect Classifiers for Each Group
    group_dialect_classifiers = train_and_save_dialect_classifiers(
        v_file=args.v_file,
        train_list=args.train_list,
        class_name=args.class_name,
        group_svm=group_svm,
        pca=pca_transform,    # Passing the trained PCA
        lnorm=lnorm_transform, # Passing the trained normalization transform
        output_dir=args.output_dir,
        verbose=args.verbose,
        dialect_groups=dialect_groups
    )
