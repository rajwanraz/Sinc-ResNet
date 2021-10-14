import yaml
import typing as tp
import os
import torchaudio.transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim
import numpy as np
import wget
from datasets.timit import TimitTrain, TimitEval

from utils import NestedNamespace, compute_chunk_info, PlotAccuracy, PlotLoss

from model.ResincNet import ResincNet
from model.MFCC import mfcc
from model.SincNet import SincNet
from model.model import SincConv


def compute_accuracy(logits: torch.Tensor, labels: tp.Union[torch.Tensor, int]) -> float:
    # 0-1
    return torch.mean((torch.argmax(logits, dim=1) == labels).float()).item()


def main(params: NestedNamespace):
    TestAcc = []
    LossPlot = []
    chunk_len, chunk_shift = compute_chunk_info(params)

    # load data train and test
    dataset_train = TimitTrain(params.data.timit.path, chunk_len=chunk_len)
    dataset_evaluation = TimitEval(
        params.data.timit.path, chunk_len, chunk_shift, 'test.scp')
    dataloader = DataLoader(dataset_train, batch_size=params.batch_size,
                            shuffle=True, drop_last=True)

    if(params.model.type == "mfcc"):
        net = mfcc(chunk_len, params.device, params.data.timit.n_classes)
    elif(params.model.type == "sinc"):
        net = SincNet(
            chunk_len, params.data.timit.n_classes,  SincConv)
    elif(params.model.type == "cnn"):
        net = SincNet(
            chunk_len, params.data.timit.n_classes,  nn.Conv1d)
    else:
        net = ResincNet()
    net = net.to(params.device)
    optim = torch.optim.RMSprop(
        net.parameters(), lr=params.lr, alpha=0.95, eps=1e-8)
    prev_epoch = 0
    if params.model.pretrain is not None:
        checkpoint = torch.load(params.model.pretrain,
                                map_location=torch.device(params.device))
        net.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epoch = checkpoint['epoch'] + 1
    criterion = nn.CrossEntropyLoss()

    for i in range(prev_epoch, prev_epoch + params.n_epochs):
        accuracy, losses = [], []
        net.train()
        for j, batch in enumerate(dataloader):
            optim.zero_grad()
            chunks, labels = batch
            chunks, labels = chunks.to(params.device), labels.to(params.device)
            logits = net(chunks)
            loss = criterion(logits, labels)
            loss.backward()
            optim.step()

            if i % params.verbose_every == 0:
                losses.append(loss.item())
                accuracy.append(compute_accuracy(logits, labels))
        print(f"epoch {i} ")

        if i % params.verbose_every == 0:
            net.eval()
            with torch.no_grad():
                chunks_accuracy, losses_test = [], []
                wavs_accuracy = 0
                for chunks, label, n_chunks in dataset_evaluation:
                    chunks = chunks.to(params.device)
                    logits = net(chunks)
                    loss = criterion(logits, torch.Tensor(
                        [label] * n_chunks).long().to(params.device))
                    losses_test.append(loss.item())
                    chunks_accuracy.append(
                        compute_accuracy(logits, label))  # 0-1
                    wavs_accuracy += (torch.argmax(logits.sum(dim=0))  # 0 or 1
                                      == label).item()
                TestAcc.append(wavs_accuracy/len(dataset_evaluation))
                LossPlot.append(np.mean(losses_test))
                print(f'epoch {i}\ntrain accuracy {np.mean(accuracy*100)}%\ntrain loss {np.mean(losses)} \n'
                      f'test loss {LossPlot[-1]}%\n'
                      f'test wav accuracy {(TestAcc[-1])*100}%')
                if len(TestAcc) > 1:
                    if TestAcc[-1] > max(TestAcc):
                        torch.save(
                            {'model_state_dict': net.state_dict(
                            ), 'optimizer_state_dict': optim.state_dict(), 'epoch': i},
                            os.path.join(params.save_path, params.model.type+'.pt'))
                        print("saved model!! ")
    PlotAccuracy(TestAcc, params.verbose_every)
    PlotLoss(LossPlot, params.verbose_every)
    print(TestAcc)
    print(LossPlot)
    f = open("TestAccLoss.txt", "w")
    f.write(f"test Acc: \n {TestAcc}\n")
    f.write(f"test Loss: \n {LossPlot}")


if __name__ == "__main__":
    with open('configs/cfg.yaml') as config:
        # wget.download("https://data.deepai.org/timit.zip")
        params = yaml.load(config, Loader=yaml.FullLoader)
        params = NestedNamespace(params)
    if params.model.type not in ['cnn', 'sinc', "mfcc", "resincNet"]:
        raise ValueError(
            "Only those models are supported, use cnn , sinc or mfcc.")
    if params.model.type == "mfcc":
        params.chunk_len_ratio = 0.4
    main(params)
