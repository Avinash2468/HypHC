"""Dataset loading."""

import os
from tqdm import tqdm
import numpy as np

UCI_DATASETS = [
    "glass",
    "zoo",
    "iris",
    "avinash",
]


def load_data(dataset, normalize=True):
    """Load dataset.

    @param dataset: dataset name
    @type dataset: str
    @param normalize: whether to normalize features or not
    @type normalize: boolean
    @return: feature vectors, labels, and pairwise similarities computed with cosine similarity
    @rtype: Tuple[np.array, np.array, np.array]
    """
    print("inside the load function")
    if dataset in UCI_DATASETS:
        x, y = load_uci_data(dataset)
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset))
    if normalize:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    print("setting x0")
    x0 = x[None, :, :].astype(float)
    print("setting x1")
    x1 = x[:, None, :].astype(float)
    
    print("THE SHAPES ARE", x0.shape, x1.shape, flush=True)
    #cos = (x0 * x1).sum(-1)
    #cos = np.multiply(x0, x1).sum(-1)
    cos = np.zeros((x0.shape[1], x0.shape[1]), dtype=float)
    #for i in tqdm(range(x0.shape[1])):
    #    cos[i,:] = (x0*x1[i]).sum(-1)
    #cos = np.einsum("ijk,mnk->jm",x0,x1)
    print("done", flush=True)
    similarities = 0.5 * (1 + cos)
    cos = None
    print("done1", flush=True)
    similarities = np.triu(similarities) 
    print("lol")
    similarities +=  np.triu(similarities).T
    print("done2", flush=True)
    similarities[np.diag_indices_from(similarities)] = 1.0
    print("done3", flush=True)
    similarities[similarities > 1.0] = 1.0
    print("finished loading", flush=True)
    return x, y, similarities


def load_uci_data(dataset):
    """Loads data from UCI repository.

    @param dataset: UCI dataset name
    @return: feature vectors, labels
    @rtype: Tuple[np.array, np.array]
    """
    x = []
    y = []
    ids = {
        "zoo": (1, 17, -1),
        "iris": (0, 4, -1),
        "glass": (1, 10, -1),
        "avinash":(1, 601, 0),
    }
    data_path = os.path.join(os.environ["DATAPATH"], dataset, "{}.data".format(dataset))
    print(data_path)
    classes = {}
    class_counter = 0
    start_idx, end_idx, label_idx = ids[dataset]
    with open(data_path, 'r') as f:
        for line in f:
            #print(line)
            line = line.strip("\n")
            split_line = line.split(",")
            
            if len(split_line) >= end_idx - start_idx + 1:
                x.append([float(x) for x in split_line[start_idx:end_idx]])
				
                label = split_line[label_idx]
                if not label in classes:
                    classes[label] = class_counter
                    class_counter += 1
                y.append(classes[label])
    print(classes.keys())
    print(len(x), len(y))
    y = np.array(y, dtype = float)
    x = np.array(x, dtype = float)
    print("INPUT SHAPES",x.shape, y.shape)
    mean = x.mean(0)
    std = x.std(0) 
    print(x)
    print("std", std)
    x = (x - mean) / std
    return x, y
