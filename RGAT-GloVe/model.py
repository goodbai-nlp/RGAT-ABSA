# coding:utf-8
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from common.tree import head_to_adj
from common.transformer_encoder import TransformerEncoder
from common.RGAT import RGATEncoder


class RGATABSA(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.hidden_dim
        self.args = args
        self.enc = ABSAEncoder(args, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(in_dim, args.num_class)

    def forward(self, inputs):
        hiddens = self.enc(inputs)
        logits = self.classifier(hiddens)
        return logits, hiddens


class ABSAEncoder(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        self.args = args
        self.emb_matrix = emb_matrix

        # #################### Embeddings ###################
        self.emb = nn.Embedding(args.tok_size, args.emb_dim, padding_idx=0)
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix.cuda(), requires_grad=False)
        self.pos_emb = (
            nn.Embedding(args.pos_size, args.pos_dim, padding_idx=0) if args.pos_dim > 0 else None
        )  # POS emb
        self.post_emb = (
            nn.Embedding(args.post_size, args.post_dim, padding_idx=0)
            if args.post_dim > 0
            else None
        )  # position emb
        
        # #################### Encoder ###################
        if self.args.model.lower() in ["std", "gat"]:
            embeddings = (self.emb, self.pos_emb, self.post_emb)
            self.encoder = DoubleEncoder(
                args, embeddings, args.hidden_dim, args.num_layers
            )
        elif self.args.model.lower() == "rgat":
            self.dep_emb = (
                nn.Embedding(args.dep_size, args.dep_dim, padding_idx=0)
                if args.dep_dim > 0
                else None
            )  # position emb
            embeddings = (self.emb, self.pos_emb, self.post_emb, self.dep_emb)
            self.encoder = DoubleEncoder(
                args, embeddings, args.hidden_dim, args.num_layers, use_dep=True
            )
        else:
            print(
                "Invalid model name {}, it should be (std, GAT, RGAT)".format(
                    self.args.model.lower()
                )
            )
            exit(0)

        # #################### pooling and fusion modules ###################
        if self.args.pooling.lower() == "attn":
            self.attn = torch.nn.Linear(args.hidden_dim, 1)

        if self.args.output_merge.lower() != "none":
            self.inp_map = torch.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        if self.args.output_merge.lower() == "none":
            pass
        elif self.args.output_merge.lower() == "attn":
            self.out_attn_map = torch.nn.Linear(args.hidden_dim * 2, 1)
        elif self.args.output_merge.lower() == "gate":
            self.out_gate_map = torch.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
        elif self.args.output_merge.lower() == "gatenorm" or self.args.output_merge.lower() == "gatenorm2":
            self.out_gate_map = torch.nn.Linear(args.hidden_dim * 2, args.hidden_dim)
            self.out_norm = nn.LayerNorm(args.hidden_dim)
        elif self.args.output_merge.lower() == "addnorm":
            self.out_norm = nn.LayerNorm(args.hidden_dim)
        else:
            print("Invalid output_merge type: ", self.args.output_merge)
            exit()

        if self.args.output_merge.lower() != "none":
            self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.eye_(self.inp_map.weight)
        torch.nn.init.zeros_(self.inp_map.bias)

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask_ori, lengths = inputs  # unpack inputs
        maxlen = max(lengths.data)

        """
        print('tok', tok, tok.size())
        print('asp', asp, asp.size())
        print('pos-tag', pos, pos.size())
        print('head', head, head.size())
        print('deprel', deprel, deprel.size())
        print('postition', post, post.size())
        print('mask', mask, mask.size())
        print('l', l, l.size())
        """
        adj_lst, label_lst = [], []
        for idx in range(len(lengths)):
            adj_i, label_i = head_to_adj(
                maxlen,
                head[idx],
                tok[idx],
                deprel[idx],
                lengths[idx],
                mask_ori[idx],
                directed=self.args.direct,
                self_loop=self.args.loop,
            )
            adj_lst.append(adj_i.reshape(1, maxlen, maxlen))
            label_lst.append(label_i.reshape(1, maxlen, maxlen))

        # [B, maxlen, maxlen]
        adj = np.concatenate(adj_lst, axis=0)
        adj = torch.from_numpy(adj).cuda()

        # [B, maxlen, maxlen]
        labels = np.concatenate(label_lst, axis=0)
        label_all = torch.from_numpy(labels).cuda()

        if self.args.model.lower() == "std":
            sent_out, graph_out = self.encoder(adj=None, inputs=inputs, lengths=lengths)
        elif self.args.model.lower() == "gat":
            sent_out, graph_out = self.encoder(adj=adj, inputs=inputs, lengths=lengths)
        elif self.args.model.lower() == "rgat":
            sent_out, graph_out = self.encoder(
                adj=adj, relation_matrix=label_all, inputs=inputs, lengths=lengths
            )
        elif self.args.model.lower() == "rgat-noadj":
            sent_out, graph_out = self.encoder(
                adj=None, relation_matrix=label_all, inputs=inputs, lengths=lengths
            )
        else:
            print(
                "Invalid model name {}, it should be (std, GAT, RGAT)".format(
                    self.args.model.lower()
                )
            )
            exit(0)

        # ###########pooling and fusion #################
        asp_wn = mask_ori.sum(dim=1).unsqueeze(-1)
        mask = mask_ori.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)  # mask for h

        if self.args.pooling.lower() == "avg":      # avg pooling
            graph_out = (graph_out * mask).sum(dim=1) / asp_wn  # masking
        elif self.args.pooling.lower() == "max":    # max pooling
            graph_out = torch.max(graph_out * mask, dim=1).values
        # elif self.args.pooling.lower() == "attn":
        #     # [B, seq_len, 1]
        #     attns = torch.tanh(self.attn(graph_out))
        #     # print('attn', attns.size())
        #     for i in range(mask_ori.size(0)):
        #         for j in range(mask_ori.size(1)):
        #             if mask_ori[i, j] == 0:
        #                 mask_ori[i, j] = -1e10
        #     masked_attns = F.softmax(mask_ori * attns.squeeze(-1), dim=1)
        #     # print('mask_attns', masked_attns.size())
        #     graph_out = torch.matmul(masked_attns.unsqueeze(1), graph_out).squeeze(1)

        if self.args.output_merge.lower() == "none":
            return graph_out

        sent_out = self.inp_map(sent_out)      # avg pooling
        if self.args.pooling.lower() == "avg":
            sent_out = (sent_out * mask).sum(dim=1) / asp_wn
        elif self.args.pooling.lower() == "max":    # max pooling
            sent_out = torch.max(sent_out * mask, dim=1).values

        if self.args.output_merge.lower() == "gate":    # gate feature fusion
            gate = torch.sigmoid(
                self.out_gate_map(torch.cat([graph_out, sent_out], dim=-1))
            )  
            outputs = graph_out * gate + (1 - gate) * sent_out
        elif self.args.output_merge.lower() == "gatenorm":
            gate = torch.sigmoid(
                self.out_gate_map(torch.cat([graph_out, sent_out], dim=-1))
            )  # gatenorm merge
            outputs = self.out_norm(graph_out * gate + (1 - gate) * sent_out)
        elif self.args.output_merge.lower() == "gatenorm2":
            gate = self.out_norm(torch.sigmoid(
                self.out_gate_map(torch.cat([graph_out, sent_out], dim=-1))
            ))  # gatenorm2 merge
            outputs = graph_out * gate + (1 - gate) * sent_out
        elif self.args.output_merge.lower() == "addnorm":
            outputs = self.out_norm(graph_out + sent_out)
        elif self.args.output_merge.lower() == "add":
            outputs = graph_out + sent_out
        elif self.args.output_merge.lower() == "attn":
            att = torch.sigmoid(
                self.out_attn_map(torch.cat([graph_out, sent_out], dim=-1))
            )  # attn merge
            outputs = graph_out * att + (1 - att) * sent_out
        return outputs


class DoubleEncoder(nn.Module):
    def __init__(self, args, embeddings, mem_dim, num_layers, use_dep=False):
        super(DoubleEncoder, self).__init__()
        self.args = args
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = args.emb_dim + args.post_dim + args.pos_dim
        if use_dep:
            self.emb, self.pos_emb, self.post_emb, self.dep_emb = embeddings
        else:
            self.emb, self.pos_emb, self.post_emb = embeddings

        # Sentence Encoder
        input_size = self.in_dim
        self.Sent_encoder = nn.LSTM(
            input_size,
            args.rnn_hidden,
            args.rnn_layers,
            batch_first=True,
            dropout=args.rnn_dropout,
            bidirectional=args.bidirect,
        )
        if args.bidirect:
            self.in_dim = args.rnn_hidden * 2
        else:
            self.in_dim = args.rnn_hidden

        # dropout
        self.rnn_drop = nn.Dropout(args.rnn_dropout)
        self.in_drop = nn.Dropout(args.input_dropout)

        # Graph Encoder
        if use_dep:
            self.graph_encoder = RGATEncoder(
                num_layers=num_layers,
                d_model=args.rnn_hidden * 2,
                heads=args.attn_heads,
                d_ff=args.rnn_hidden * 2,
                dropout=args.layer_dropout,
                att_drop=args.att_dropout,
                use_structure=True,
                alpha=args.alpha,
                beta=args.beta,
            )
        else:
            self.graph_encoder = TransformerEncoder(
                num_layers=num_layers,
                d_model=args.rnn_hidden * 2,
                heads=args.attn_heads,
                d_ff=args.rnn_hidden * 2,
                dropout=args.layer_dropout,
            )

        self.out_map = nn.Linear(args.rnn_hidden * 2, args.rnn_hidden)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(
            batch_size, self.args.rnn_hidden, self.args.rnn_layers, self.args.bidirect
        )
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.Sent_encoder(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs, lengths, relation_matrix=None):
        tok, asp, pos, head, deprel, post, a_mask, seq_len = inputs  # unpack inputs
        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # Sentence encoding
        sent_output = self.rnn_drop(
            self.encode_with_rnn(embs, seq_len, tok.size()[0])
        )  # [B, seq_len, H]

        mask = adj.eq(0) if adj is not None else None
        key_padding_mask = sequence_mask(lengths) if lengths is not None else None  # [B, seq_len]
        dep_relation_embs = self.dep_emb(relation_matrix) if relation_matrix is not None else None

        # Graph encoding 
        inp = sent_output
        graph_output = self.graph_encoder(
            inp, mask=mask, src_key_padding_mask=key_padding_mask, structure=dep_relation_embs,
        )  # [bsz, seq_len, H]
        graph_output = self.out_map(graph_output)
        return sent_output, graph_output


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size, 1, seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))
