import os, random
import argparse
import numpy as np

from sklearn import metrics
from scipy import interpolate
from scipy.optimize import brentq
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
from copy import deepcopy

import matplotlib
matplotlib.use('Agg', force=True)
#matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt


def draw_roc(y_test, y_pred_prob, name=None):
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
    name='DTPV'
    plt.clf()
    fig = plt.figure(1)
    plt.plot(fpr, tpr, 'r')
    plt.xlim([0.0, 0.01])
    plt.ylim([0.90, 1.0])
    plt.title(f' ROC curve of {name}')
    plt.xlabel('FPR (False Positive Rate)')
    plt.ylabel('TPR (True Positive Rate)')
    plt.grid(True)
    plt.savefig(f'ROC_{name}.png')
    plt.close(fig)


def roc(feature, nproc=8):
    genuine_scores = Intra_Score_Process(feature, nproc=nproc)
    imposter_scores = Inter_Score_Process(feature, nproc=nproc)

    genuine_label = np.ones(len(genuine_scores), dtype=np.int32).tolist()
    imposter_label = np.zeros(len(imposter_scores), dtype=np.int32).tolist()

    # calculate label and prob
    label = genuine_label + imposter_label
    prob = genuine_scores + imposter_scores
    draw_roc(label, prob)

def calculate_performance(y_test, y_pred_prob, return_curve=False):

    # ROC
    # IMPORTANT: first argument is true values, second argument is predicted probabilities
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

    # AUC
    AUC = metrics.auc(fpr, tpr)

    # EER
    EER = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)

    # TAR @ FAR = 0.1 / 0.01 / 0.001, FAR = FPR, TAR = TPR
    TAR_FAR_E1 = brentq(lambda x: 0.1 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E2 = brentq(lambda x: 0.01 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E3 = brentq(lambda x: 0.001 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    TAR_FAR_E4 = brentq(lambda x: 0.0001 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    try:
        TAR_FAR_E5 = brentq(lambda x: 0.00001 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    except:
        TAR_FAR_E5 = 0

    try:
        TAR_FAR_E6 = brentq(lambda x: 0.000001 - interpolate.interp1d(tpr, fpr)(x), 0., 1.)
    except:
        TAR_FAR_E6 = 0
    # return ACC, AUC, TAR_FAR_E1, TAR_FAR_E2, TAR_FAR_E3, fpr, tpr
    results = {
        "AUC":AUC,
        "EER": EER,
        "TAR_FAR_E1": TAR_FAR_E1,
        "TAR_FAR_E2": TAR_FAR_E2,
        "TAR_FAR_E3": TAR_FAR_E3,
        "TAR_FAR_E4": TAR_FAR_E4,
        "TAR_FAR_E5": TAR_FAR_E5,
        "TAR_FAR_E6": TAR_FAR_E6,
    }
    if return_curve:
        roc_curve = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }
        return results, roc_curve
    return results


def save_roc_data(path, genuine_scores, imposter_scores, roc_curve):
    filename = path + '_roc_data.npz'
    np.savez_compressed(
        filename,
        genuine_scores=np.asarray(genuine_scores, dtype=np.float32),
        imposter_scores=np.asarray(imposter_scores, dtype=np.float32),
        fpr=np.asarray(roc_curve["fpr"], dtype=np.float32),
        tpr=np.asarray(roc_curve["tpr"], dtype=np.float32),
        thresholds=np.asarray(roc_curve["thresholds"], dtype=np.float32),
    )
    return filename


def calculate_cosine_similarity(array1, array2):
    # # Normalize the arrays 转为单位向量
    # normalized_array1 = array1  
    # normalized_array2 = array2  

    # # Calculate cosine similarity  计算余弦相似度
    similarity_matrix = cosine_similarity(array1, array2)

    similarity_matrix = cosine_similarity(array1, array2)
    # similarity_matrix = np.dot(array1, array2.T)

    return similarity_matrix


def load_file(filepath):
    data_path = filepath + '_feature.npz'
    name_path = filepath + '_name.CASIA'

    # load features
    data = np.load(data_path)["features"]
    namelist = open(name_path).readlines()
    id_list = [ int(name.split(' ')[-1]) for name in namelist]
    feats_list = [[] for i in range(sorted(id_list)[-1] + 1)]

    # store features according to IDs
    [feats_list[i].append(f[None]) for i, f in zip(id_list, data)]
    for i in range(len(feats_list)):
        feats_list[i] = np.concatenate(feats_list[i], axis=0)

    return feats_list, id_list


def Intra_Score(features: list) -> list:
    results = []
    for i in range(len(features)):
        data = features[i]  # data.shape == (60, 512)
        idx = np.tril_indices(data.shape[0], -1)
        similarity_matrix = calculate_cosine_similarity(data, data)
        sim = similarity_matrix[idx].tolist()
        results += sim
    return results


def cal_intra_score_task(idx: int, features: list, idx_list: list, results_list: list) -> list:
    # print(f"cal intra process-{idx}")
    results = []
    for i in idx_list:
        data = features[i]  
        idx_array = np.tril_indices(data.shape[0], -1)
        similarity_matrix = calculate_cosine_similarity(data, data)
        sim = similarity_matrix[idx_array].tolist()
        results += sim
    # results_list[idx] = results
    # return results
    results_list.put(results)

    # print(f"cal intra process-{idx} is done")



def Intra_Score_Process(features: list, nproc: int = 8) -> list:
    
    queue = multiprocessing.Manager().Queue()

    idx_list = [i for i in range(len(features))]
    process_list = []
    for i in range(nproc):
        process_list.append(multiprocessing.Process(target=cal_intra_score_task, args=(i, features, idx_list[i::nproc], queue)))

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()
    
    results = []
    while not queue.empty():
        results += queue.get()
    return results



def Inter_Score(features: list, ratio: float=1.0) -> list:
    # reduced_features = random.choices(features, k=int(len(features)*ratio))
    reduced_features = features
    results = []    
    for i in range(len(reduced_features)-1):
        for j in range(i+1, len(reduced_features)):
            feats1 = reduced_features[i]
            feats2 = reduced_features[j]
            similarity_matrix = calculate_cosine_similarity(feats1, feats2)
            p = similarity_matrix.flatten().tolist()
            results += p
    return results
    

def cal_inter_score_task(idx: int, features: list, idx_list: list, results_list: list):
    # print(f"cal inter process-{idx}")
    results = []
    for i, j in idx_list:
        feats1 = features[i]
        feats2 = features[j]
        similarity_matrix = calculate_cosine_similarity(feats1, feats2)
        p = similarity_matrix.flatten().tolist()
        results += p 
    # results_list[idx] = results
    results_list.put(results)


def Inter_Score_Process(features: list, nproc: int = 8) -> list:
    
    # results_list = [[] for i in range(nproc)]
    queue = multiprocessing.Manager().Queue()
    idx_list = []
    for i in range(len(features)-1):
        for j in range(i+1, len(features)):
            idx_list.append((i, j))
    process_list = []
    for i in range(nproc):
        process_list.append(multiprocessing.Process(target=cal_inter_score_task, args=(i, features, idx_list[i::nproc], queue)))

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()
    
    results = []
    while not queue.empty():
        results += queue.get()
    return results


def score_distribution(feature, name):
    genuine_scores = Intra_Score(feature)
    imposter_scores = Inter_Score(feature, ratio=0.2)

    plt.clf()
    plt.hist(genuine_scores, bins=np.arange(-1, 1, 0.01), label="genuine", color="green", alpha=0.5, density=True)
    plt.hist(imposter_scores, bins=np.arange(-1, 1, 0.01), label="imposter", color="red", alpha=0.5, density=True)
    plt.xlim(-1, 1)
    plt.yticks([])
    plt.ylabel('probability')
    plt.rcParams.update({'font.size': 17})
    plt.legend(loc='upper left')
    plt.title('Score distribution of '+f'{name}')
    plt.savefig(f"{name}_distributions.png", dpi=512)

    genuine_scores = np.array(genuine_scores)
    imposter_scores = np.array(imposter_scores)

    print(f'intra stat {name} -> mean: {np.mean(genuine_scores)};  std: {np.std(genuine_scores)}')
    print(f'inter stat {name} -> mean: {np.mean(imposter_scores)};  std: {np.std(imposter_scores)}')


def calculate_eer(feature, path=None):

    genuine_scores = Intra_Score(feature)
    imposter_scores = Inter_Score(feature, ratio=0.2)

    genuine_label = np.ones(len(genuine_scores), dtype=np.int32).tolist()
    imposter_label = np.zeros(len(imposter_scores), dtype=np.int32).tolist()

    # calculate label and prob
    label = genuine_label + imposter_label
    prob = genuine_scores + imposter_scores

    results, roc_curve = calculate_performance(label, prob, return_curve=True)

    if path is not None:
        filename = path + '_results.CASIA'
        with open(filename, "w") as f:
            for k, v in results.items():
                Line = f"{k} --> {v}\n"
                f.write(Line)        
        roc_filename = save_roc_data(path, genuine_scores, imposter_scores, roc_curve)
        print(f"roc_data --> {roc_filename}")
        print('-' * 20 + f'  {os.path.basename(path)}  ' + '-' * 20)
    else:
        print('-' * 20 + '-' * 20)
    for k, v in results.items():
        print(f"{k} --> {v}")
    return results


def calculate_eer_multi(feature, path=None, nproc=64):

    genuine_scores = Intra_Score_Process(feature, nproc=nproc)
    imposter_scores = Inter_Score_Process(feature, nproc=nproc)

    genuine_label = np.ones(len(genuine_scores), dtype=np.int32).tolist()
    imposter_label = np.zeros(len(imposter_scores), dtype=np.int32).tolist()

    # calculate label and prob
    label = genuine_label + imposter_label
    prob = genuine_scores + imposter_scores

    results, roc_curve = calculate_performance(label, prob, return_curve=True)

    if path is not None:
        filename = path + '_results.txt'
        with open(filename, "w") as f:
            for k, v in results.items():
                Line = f"{k} --> {v}\n"
                f.write(Line)        
        roc_filename = save_roc_data(path, genuine_scores, imposter_scores, roc_curve)
        print(f"roc_data --> {roc_filename}")
        print('-' * 20 + f'  {os.path.basename(path)}  ' + '-' * 20)
    else:
        print('-' * 20 + '-' * 20)
    for k, v in results.items():
        print(f"{k} --> {v}")
    return results


def score_distribution_multi(feature, name, path, nproc=4):
    genuine_scores = Intra_Score_Process(feature, nproc=nproc)
    imposter_scores = Inter_Score_Process(feature, nproc=nproc)

    plt.clf()
    plt.figure()
    plt.hist(genuine_scores, bins=np.arange(-1, 1, 0.001), label="genuine", color="green", alpha=0.5, density=True)
    plt.hist(imposter_scores, bins=np.arange(-1, 1, 0.001), label="imposter", color="red", alpha=0.5, density=True)
    plt.xlim(-1, 1)
    # plt.ylim(0, 6.5)
    plt.legend()
    plt.title(f'{name}')
    plt.savefig(f"{path}", dpi=512)
    plt.close()

    genuine_scores = np.array(genuine_scores)
    imposter_scores = np.array(imposter_scores)

    print(f'intra stat {name} -> mean: {np.mean(genuine_scores)};  std: {np.std(genuine_scores)}')
    print(f'inter stat {name} -> mean: {np.mean(imposter_scores)};  std: {np.std(imposter_scores)}')

    results = {
        "intra-mean":np.mean(genuine_scores),
        "intra-std":np.std(genuine_scores),
        "inter-mean":np.mean(imposter_scores),
        "inter-std":np.std(imposter_scores)
    }

    return results



if __name__ == '__main__':
    features = [np.random.randn(10, 512) for i in range(5)]
    intra_score = np.array(Intra_Score(features))
    inter_score = np.array(Inter_Score(features, ratio=0.2))
    intra_score_2 = np.array(Intra_Score_Process(features, nproc=8))
    inter_score_2 = np.array(Inter_Score_Process(features, nproc=8))

    print(np.sort(intra_score) == np.sort(intra_score_2))
    print("="*100)
    print(np.sort(inter_score) == np.sort(inter_score_2))
