# coding:utf-8
import sys
sys.path.append('../')
import torch
import numpy as np
import torch.nn as nn
from common.tree import head_to_adj
from common.transformer_encoder import TransformerEncoder
from common.RGAT import RGATEncoder
from transformers import BertModel, BertConfig

bert_config = BertConfig.from_pretrained("bert-base-uncased")
bert_config.output_hidden_states = True
bert_config.num_labels = 3
bert = BertModel.from_pretrained("bert-base-uncased", config=bert_config)


class RGATABSA(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super().__init__()
        in_dim = args.hidden_dim + args.bert_out_dim
        self.args = args
        self.enc = ABSAEncoder(args)
        self.classifier = nn.Linear(in_dim, args.num_class)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        outputs = self.enc(inputs)
        outputs = self.dropout(outputs)
        logits = self.classifier(outputs)
        return logits, outputs


class ABSAEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pos_emb = (
            nn.Embedding(args.pos_size, args.pos_dim, padding_idx=0) if args.pos_dim > 0 else None
        )  # pos tag emb
        self.post_emb = (
            nn.Embedding(args.post_size, args.post_dim, padding_idx=0)
            if args.post_dim > 0
            else None
        )  # position emb
        if self.args.model.lower() in ["std", "gat"]:
            embs = (self.pos_emb, self.post_emb)
            self.encoder = DoubleEncoder(args, embeddings=embs, use_dep=True)
        elif self.args.model.lower() == "rgat":
            self.dep_emb = (
                nn.Embedding(args.dep_size, args.dep_dim, padding_idx=0)
                if args.dep_dim > 0
                else None
            )  # position emb
            embs = (self.pos_emb, self.post_emb, self.dep_emb)
            self.encoder = DoubleEncoder(args, embeddings=embs, use_dep=True)

        if self.args.output_merge.lower() == "gate":
            self.gate_map = nn.Linear(args.bert_out_dim * 2, args.bert_out_dim)
        elif self.args.output_merge.lower() == "none":
            pass
        else:
            print('Invalid output_merge type !!!')
            exit()

    def forward(self, inputs):
        (
            tok,
            asp,
            pos,
            head,
            deprel,
            post,
            mask,
            l,
            text_raw_bert_indices,
            bert_sequence,
            bert_segments_ids,
        ) = inputs  # unpack inputs
        maxlen = max(l.data)
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
        for idx in range(len(l)):
            adj_i, label_i = head_to_adj(
                maxlen,
                head[idx],
                tok[idx],
                deprel[idx],
                l[idx],
                mask[idx],
                directed=self.args.direct,
                self_loop=self.args.loop,
            )
            adj_lst.append(adj_i.reshape(1, maxlen, maxlen))
            label_lst.append(label_i.reshape(1, maxlen, maxlen))

        adj = np.concatenate(adj_lst, axis=0)  # [B, maxlen, maxlen]
        adj = torch.from_numpy(adj).cuda()

        labels = np.concatenate(label_lst, axis=0)  # [B, maxlen, maxlen]
        label_all = torch.from_numpy(labels).cuda()
        if self.args.model.lower() == "std":
            h = self.encoder(adj=None, inputs=inputs, lengths=l)
        elif self.args.model.lower() == "gat":
            h = self.encoder(adj=adj, inputs=inputs, lengths=l)
        elif self.args.model.lower() == "rgat":
            h = self.encoder(
                adj=adj, relation_matrix=label_all, inputs=inputs, lengths=l
            )
        else:
            print(
                "Invalid model name {}, it should be (std, GAT, RGAT)".format(
                    self.args.model.lower()
                )
            )
            exit(0)

        graph_out, bert_pool_output, bert_out = h[0], h[1], h[2]
        asp_wn = mask.sum(dim=1).unsqueeze(-1)                          # aspect words num
        mask = mask.unsqueeze(-1).repeat(1, 1, self.args.bert_out_dim)  # mask for h
        graph_enc_outputs = (graph_out * mask).sum(dim=1) / asp_wn        # mask h
        bert_enc_outputs = (bert_out * mask).sum(dim=1) / asp_wn

        if self.args.output_merge.lower() == "none":
            merged_outputs = graph_enc_outputs
        elif self.args.output_merge.lower() == "gate":
            gate = torch.sigmoid(self.gate_map(torch.cat([graph_enc_outputs, bert_enc_outputs], 1)))
            merged_outputs = gate * graph_enc_outputs + (1 - gate) * bert_enc_outputs
        else:
            print('Invalid output_merge type !!!')
            exit()
        cat_outputs = torch.cat([merged_outputs, bert_pool_output], 1)
        return cat_outputs


class DoubleEncoder(nn.Module):
    def __init__(self, args, embeddings=None, use_dep=False):
        super(DoubleEncoder, self).__init__()
        self.args = args
        self.Sent_encoder = bert
        self.in_drop = nn.Dropout(args.input_dropout)
        self.dense = nn.Linear(args.hidden_dim, args.bert_out_dim)  # dimension reduction
        
        if use_dep:
            self.pos_emb, self.post_emb, self.dep_emb = embeddings
            self.Graph_encoder = RGATEncoder(
                num_layers=args.num_layer,
                d_model=args.bert_out_dim,
                heads=4,
                d_ff=args.hidden_dim,
                dep_dim=self.args.dep_dim,
                att_drop=self.args.att_dropout,
                dropout=0.0,
                use_structure=True
            )
        else:
            self.pos_emb, self.post_emb = embeddings
            self.Graph_encoder = TransformerEncoder(
                num_layers=args.num_layer,
                d_model=args.bert_out_dim,
                heads=4,
                d_ff=args.hidden_dim,
                dropout=0.0
            )
        if args.reset_pooling:
            self.reset_params(bert.pooler.dense)

    def reset_params(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    def forward(self, adj, inputs, lengths, relation_matrix=None):
        (
            tok,
            asp,
            pos,
            head,
            deprel,
            post,
            a_mask,
            l,
            text_raw_bert_indices,
            bert_sequence,
            bert_segments_ids,
        ) = inputs  # unpack inputs

        bert_sequence = bert_sequence[:, 0:bert_segments_ids.size(1)]
        # input()
        bert_out, bert_pool_output, bert_all_out = self.Sent_encoder(
            bert_sequence, token_type_ids=bert_segments_ids
        )
        bert_out = self.in_drop(bert_out)
        bert_out = bert_out[:, 0:max(l), :]
        bert_out = self.dense(bert_out)

        if adj is not None:
            mask = adj.eq(0)
        else:
            mask = None
        # print('adj mask', mask, mask.size())
        if lengths is not None:
            key_padding_mask = sequence_mask(lengths)  # [B, seq_len]
        
        if relation_matrix is not None:
            dep_relation_embs = self.dep_emb(relation_matrix)
        else:
            dep_relation_embs = None

        inp = bert_out  # [bsz, seq_len, H]
        graph_out = self.Graph_encoder(
            inp, mask=mask, src_key_padding_mask=key_padding_mask, structure=dep_relation_embs
        )               # [bsz, seq_len, H]
        return graph_out, bert_pool_output, bert_out


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size, 1, seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))

