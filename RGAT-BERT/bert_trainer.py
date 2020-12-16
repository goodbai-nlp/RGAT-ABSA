import torch
import torch.nn.functional as F
import numpy as np

from transformers import AdamW
from bert_model import RGATABSA

from utils import torch_utils


class ABSATrainer(object):
    def __init__(self, args, emb_matrix=None):
        self.args = args
        self.emb_matrix = emb_matrix
        self.model = RGATABSA(args)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.model.cuda()
        self.optimizer = torch_utils.get_optimizer(args.optim, self.parameters, args.lr, l2=1e-5)
        # '''
        bert_model = self.model.enc.encoder.Sent_encoder
        bert_params_dict = list(map(id, bert_model.parameters()))
        base_params = filter(lambda p: id(p) not in bert_params_dict, self.model.parameters())
        # no_decay = ["bias", "LayerNorm.weight"]
        # optimizer_grouped_parameters = [
        #    {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": args.l2,},
        #    {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": args.l2},
        #    {"params": base_params},
        #    {"params": bert_model.parameters(), "lr": args.bert_lr}
        # ]
        optimizer_grouped_parameters = [
            {"params": base_params},
            {"params": bert_model.parameters(), "lr": args.bert_lr},
        ]
        self.optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=args.lr, weight_decay=args.l2
        )
        # self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
        # '''

    # load model_state and args
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint["model"])
        self.args = checkpoint["config"]

    # save model_state and args
    def save(self, filename):
        params = {
            "model": self.model.state_dict(),
            "config": self.args,
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
        inputs = batch[0:11]
        label = batch[-1]

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, _ = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction="mean")
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
        inputs = batch[0:11]
        label = batch[-1]

        # forward
        self.model.eval()
        logits, g_outputs = self.model(inputs)
        loss = F.cross_entropy(logits, label, reduction="mean")
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        predprob = F.softmax(logits, dim=1).data.cpu().numpy().tolist()

        return (
            loss.data,
            acc,
            predictions,
            label.data.cpu().numpy().tolist(),
            predprob,
            g_outputs.data.cpu().numpy(),
        )