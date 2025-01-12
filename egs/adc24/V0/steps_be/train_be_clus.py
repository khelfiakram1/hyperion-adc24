#!/usr/bin/env python
"""
Clustering x-vectors and training SVM classifiers for dialect classification.
"""
from collections import defaultdict
import sys
import os
import logging
from jsonargparse import ArgumentParser, namespace_to_dict, ActionYesNo
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from hyperion.hyp_defs import config_logger
from hyperion.utils import SegmentSet
from hyperion.io import RandomAccessDataReaderFactory as DRF
from hyperion.np.transforms import TransformList, PCA, LNorm
from hyperion.np.classifiers import LinearSVMC as SVM
from hyperion.np.clustering import KMeans as KM
from hyperion.np.clustering import KMeansInitMethod
from hyperion.np.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    print_confusion_matrix,
)
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def compute_metrics(y_true, y_pred, labels):
    acc = compute_accuracy(y_true, y_pred)
    logging.info("training acc: %.2f %%", acc * 100)
    logging.info("non-normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=False)
    print_confusion_matrix(C, labels)
    logging.info("normalized confusion matrix:")
    C = compute_confusion_matrix(y_true, y_pred, normalize=True)
    print_confusion_matrix(C * 100, labels)


def train_clustering_and_classification(
    v_file,
    train_list,
    class_name,
    do_lnorm,
    whiten,
    pca,
    num_clusters,
    svm,
    output_dir,
    verbose,
):
    config_logger(verbose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("loading data")
    train_segs = SegmentSet.load(train_list)
    train_reader = DRF.create(v_file)
    x_trn = train_reader.read(train_segs["id"], squeeze=True)
    del train_reader
    class_ids = train_segs[class_name]
    labels, y_true = np.unique(class_ids, return_inverse=True)
    logging.info("loaded %d samples", x_trn.shape[0])

    # Step 1: PCA
    logging.info("PCA args=%s", str(pca))
    pca_dim = pca["pca_dim"]
    if pca_dim is not None:
        logging.info("training PCA")
        pca = PCA(**pca)
        pca.fit(x_trn)
        logging.info("PCA dimension: %d", pca.pca_dim)
        logging.info("apply PCA")
        x_trn = pca(x_trn)
    else:
        pca = None

    # Step 2: Normalize
    if do_lnorm:
        lnorm = LNorm()
        if whiten:
            logging.info("training whitening")
            lnorm.fit(x_trn)
        logging.info("apply lnorm")
        x_trn = lnorm(x_trn)
    else:
        lnorm = None

    # Step 3: K-Means Clustering
    logging.info("Performing K-Means clustering with %d clusters", num_clusters)
    
    
    dialect_mean_vectors = []
    logging.info("Computing mean vectors for each dialect...")

    for label in np.unique(y_true):
        dialect_xvectors = x_trn[y_true == label]  # Get x-vectors for this dialect
        mean_vector = np.mean(dialect_xvectors, axis=0)  # Mean vector
        dialect_mean_vectors.append(mean_vector)

    # Convert to NumPy array
    dialect_mean_vectors = np.array(dialect_mean_vectors)

    logging.info(f"Clustering the {len(labels)} dialects into {num_clusters} groups...")
    kmeans = KM(num_clusters=num_clusters)

    loss, cluster_ids = kmeans.fit(dialect_mean_vectors)
    for dialect, cluster in zip(labels, cluster_ids):
        logging.info(f"Dialect: {dialect}, Cluster: {cluster}")

    dialects_groups = defaultdict(list)
    for dialect, cluster in zip(labels, cluster_ids):
        dialects_groups[cluster].append(dialect)

    dialects_groups = dict(dialects_groups)
    # for cluster_id, dialects in dialects_groups.items():

    #     # if cluster_id != 0 :
    #     #     continue
    #     logging.info(f"Training classifier for cluster: {cluster_id}")
    #     filtered_train_segs = train_segs.df[train_segs.df[class_name].isin(dialects)]
    #     filtered_train_segs = SegmentSet(df=filtered_train_segs)
    #     group_output_dir = output_dir / f"cluster_{cluster_id}"
    #     group_output_dir.mkdir(parents=True, exist_ok=True)
    #     train_reader = DRF.create(v_file)
    #     x_trn = train_reader.read(filtered_train_segs["id"], squeeze=True)
    #     del train_reader
    #     class_ids = filtered_train_segs[class_name]
    #     labels, y_true = np.unique(class_ids, return_inverse=True)
    #     logging.info(f"Loaded {x_trn.shape[0]} samples for cluster {cluster_id}")
    #     pca_transform = None
    #     if do_lnorm:
    #         lnorm = LNorm()
    #         if whiten:
    #             logging.info("training whitening for group %s",cluster_id)
    #             lnorm.fit(x_trn)
    #         logging.info("apply lnorm for group %s", cluster_id)
    #         x_trn = lnorm(x_trn)
    #     else:
    #         lnorm = None

    #     logging.info("SVM args=%s", str(svm))
    #     model = SVM(labels=labels, **svm)
    #     model.fit(x_trn, y_true)
    #     logging.info("trained SVM for group %s", cluster_id)


    #     scores = model(x_trn)
    #     y_pred = np.argmax(scores, axis=-1)
    #     compute_metrics(y_true, y_pred, labels)
        
    #     logging.info("Saving transforms and SVM")
    #     transforms = []
    #     if pca is not None:
    #         transforms.append(pca)
    #     if lnorm is not None:
    #         transforms.append(lnorm)

    #     if transforms:
    #         transforms = TransformList(transforms)
    #         transforms.save(output_dir / "transforms.h5")

    #     model.save(group_output_dir / "groups_model_svm.h5")
    # kmeans.save_model(output_dir / "cluster.h5")
    #------------------------------------------------
    # Step 4: Train SVM for each cluster
    # classifiers = {}
    # for cluster_id in range(num_clusters):
    #     cluster_idx = np.where(cluster_ids == cluster_id)
    #     x_cluster = x_trn[cluster_idx]
    #     y_cluster = y_true[cluster_idx]

    #     logging.info("Training SVM for cluster %d", cluster_id)
    #     model = SVM(labels=labels, **svm)
    #     model.fit(x_cluster, y_cluster)

    #     classifiers[cluster_id] = model

    #     # Evaluate SVM on the cluster data
    #     scores = model(x_cluster)
    #     y_pred = np.argmax(scores, axis=-1)
    #     compute_metrics(y_cluster, y_pred, labels)

    # # Save the clustering model, transforms, and SVM classifiers
    # logging.info("Saving clustering model, transforms, and SVM models")
    # kmeans_model_path = output_dir / "kmeans_model.pkl"
    # np.save(kmeans_model_path, kmeans)

    # transforms = []
    # if pca is not None:
    #     transforms.append(pca)
    # if lnorm is not None:
    #     transforms.append(lnorm)
    # if transforms:
    #     transforms = TransformList(transforms)
    #     transforms.save(output_dir / "transforms.h5")

    # for cluster_id, model in classifiers.items():
    #     model_path = output_dir / f"svm_model_cluster_{cluster_id}.h5"
    #     model.save(model_path)

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Deep clustering of x-vectors and training SVM classifiers"
    )

    parser.add_argument("--v-file", required=True, help="Path to x-vector file")
    parser.add_argument("--train-list", required=True, help="Path to train list")
    parser.add_argument("--class-name", default="class_id", help="Class label field")
    parser.add_argument("--num-clusters", type=int, required=True, help="Number of clusters for K-Means")
    PCA.add_class_args(parser, prefix="pca")
    SVM.add_class_args(parser, prefix="svm")
    #KM.add_class_args(parser, prefix="km")
    parser.add_argument("--do-lnorm", default=True, action=ActionYesNo)
    parser.add_argument("--whiten", default=True, action=ActionYesNo)
    parser.add_argument("--output-dir", required=True, help="Directory to save models")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", default=1, choices=[0, 1, 2, 3], type=int
    )

    args = parser.parse_args()
    train_clustering_and_classification(**namespace_to_dict(args))
