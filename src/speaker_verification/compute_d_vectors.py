from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import torch

from datasets.timit import TimitEval
from utils import compute_chunk_info, get_params, load_model

parser = ArgumentParser()
parser.add_argument('model_type', help="should be cnn or sinc")
parser.add_argument('pretrained_model', help="path to pretrained model")
parser.add_argument('compute_split',
                    help='should be sv.scp or train.scp for unseen and default d-vectors respectively')
parser.add_argument('--save_to', default='d_vectors_random.npy')
args = parser.parse_args()

params = get_params('configs/cfg_sv.yaml')

chunk_len, chunk_shift = compute_chunk_info(params)
sinc_net = load_model(params, args, chunk_len)

d_vectors = defaultdict(list)
d_vectors_final = {}
evaluation_test = TimitEval(params.data.timit.path, chunk_len, chunk_shift, args.compute_split)
with torch.no_grad():
    for chunks, label, n_chunks in evaluation_test:
        d_vectors_chunk = sinc_net.compute_d_vectors(chunks.to(params.device))
        cur_d_vector = (d_vectors_chunk / d_vectors_chunk.norm(p=2, dim=1, keepdim=True)).mean(dim=0)
        d_vectors[label].append(cur_d_vector.cpu().numpy())
        if (len(d_vectors[label]) == 3 and args.compute_split == 'sv.scp') \
                or (len(d_vectors[label]) == 5 and args.compute_split == 'train.scp'):
            d_vectors_final[label] = np.mean(d_vectors[label], axis=0)

np.save(args.save_to, d_vectors_final)
