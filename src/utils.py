import typing as tp
from types import SimpleNamespace

import torch
import yaml
import matplotlib.pyplot as plt
from model.SincNet import SincNet


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)


def compute_chunk_info(params: NestedNamespace) -> tp.Tuple[int, int]:
    chunk_len = int(params.sample_rate * params.chunk_len_ratio)
    chunk_shift = int(params.sample_rate * params.chunk_shift_ratio)
    return chunk_len, chunk_shift


def get_params(path_to_cfg: str):
    with open(path_to_cfg) as config:
        params = yaml.load(config, Loader=yaml.FullLoader)
        params = NestedNamespace(params)
    return params


def load_model(params, args, chunk_len):
    sinc_net = SincNet(chunk_len, params.data.timit.n_classes, args.model_type)
    checkpoint = torch.load(args.pretrained_model,
                            map_location=torch.device(params.device))
    sinc_net.load_state_dict(checkpoint['model_state_dict'])
    sinc_net = sinc_net.to(params.device)
    sinc_net.eval()
    return sinc_net


def PlotAccuracy(Accuracy, verbose_every):
    x = [x for x in range(0, len(Accuracy)
                          * verbose_every, verbose_every)]
    y = [y*100 for y in Accuracy]
    plt.plot(x, y, label='Accuracy', color='green')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy [%]')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.title('Accuracy')
    path = "currentAccuracy.png"
    plt.savefig(path)
    plt.show()
    plt.close()


def PlotLoss(loss, verbose_every):
    x = [x for x in range(0,
                          len(loss)*verbose_every, verbose_every)]
    y = [y*100 for y in loss]
    plt.plot(x, y, label='Loss', color='red')
    plt.xlabel('epoch')
    plt.ylabel('Loss [%]')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.title('Loss')
    path = "currentLoss.png"
    plt.savefig(path)
    plt.show()
    plt.close()
