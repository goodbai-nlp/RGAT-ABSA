# coding:utf-8
import torch
import torch.nn.functional as F
import numpy as np

from model import RGATABSA
from utils import torch_utils


class ABSATrainer(object):
    def __init__(self, args, emb_matrix=None):
        self.args = args
        self.emb_matrix = emb_matrix
        self.model = RGATABSA(args, emb_matrix=emb_matrix)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.model.cuda()
        self.optimizer = torch_utils.get_optimizer(args.optim, self.parameters, args.lr)

    # load model_state and args
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.args = checkpoint['config']

    # save model_state and args
    def save(self, filename):
        params = {
            'model': self.model.state_dict(),
            'config': self.args,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def update(self, batch):
        # convert to cuda
        batch = [b.cuda() for b in batch]

        # unpack inputs and label
        inputs = batch[0:8]
        label = batch[-1]

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, _ = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction='mean')
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        
        # backward
        loss.backward()
        self.optimizer.step()
        return loss.data, acc

    def predict(self, batch):
        # convert to cuda
        batch = [b.cuda() for b in batch]

        # unpack inputs and label
        inputs = batch[0:8]
        label = batch[-1]

        # forward
        self.model.eval()
        logits, g_outputs = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction='mean')
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        predprob = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        
        return loss.data, acc, predictions, label.data.cpu().numpy().tolist(), predprob, g_outputs.data.cpu().numpy()

    def show_error(self, batch, vocab=None):
        # convert to cuda
        batch = [b.cuda() for b in batch]

        # unpack inputs and label
        inputs = batch[0:8]
        label = batch[-1]

        # forward
        self.model.eval()
        logits, g_outputs = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction='mean')
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        # wrongs = (torch.max(logits, 1)[1].view(label.size()).data != label.data)
        # print('batch', batch)
        # print('run error', wrongs)
        acc = 100.0 * np.float(corrects) / label.size()[0]
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        
        # print('acc', acc)
        # print('predictions', predictions)
        predprob = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        
        for i in range(len(batch)):
            tokids = batch[0][i]
            aspids = batch[1][i]
            ithlabel = batch[-1][i]
            pridict = predictions[i] 
            if vocab is not None:
                # print(tokids)
                tok = [vocab.itos[idx] for idx in tokids]
                asp_tok = [vocab.itos[idx] for idx in aspids]
                # strline = ' '.join(tok) + ' '.join(asp_tok) + str(label.item()) + str(pridict.item())
                # if ithlabel.item() != pridict:
                strline = '{} {} {} {} {}'.format(' '.join(tok), ' '.join(asp_tok), ithlabel.item(), pridict, ithlabel.item() == pridict)
                print(strline)
        
        return loss.data, acc, predictions, label.data.cpu().numpy().tolist(), predprob, g_outputs.data.cpu().numpy()

