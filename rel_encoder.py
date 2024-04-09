import torch
import torch.nn as nn
from opt_einsum import contract
from long_seq import process_long_input

class Rel_encoder(nn.Module):
    def __init__(self, config, model, num_class, low_dim,emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, num_class)


        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        # sequence_output [batch_size,seq_len,emb_size]  4*360*768 
        # attention [batch_size,h,seq_len,seq_len] 4*12*360*360
        # entity_pos 实体位置
        # hts 候选实体对
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset]) #emb_size 768
                            e_att.append(attention[i, :, start + offset]) # [h,seq_len]  12*360
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0) #[h,seq_len] 12*360,对多个实体的注意力求平均
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, emb_size]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device) #所有实体对
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0]) #[n_hti, emb_size]
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1]) #[n_hti, emb_size]

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0]) #[n_hti, h, seq_len]
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1]) #[n_hti, h, seq_len]
            ht_att = (h_att * t_att).mean(1) #对应位置点乘,求所有注意力头的平均 [n_hti,seq_len]
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)# 归一化处理
            rs = contract("ld,rl->rd", sequence_output[i], ht_att) #矩阵相乘[n_hti,emb_size]
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)
        ##[n_hti,emb_size]

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size) #[n_hti, self.emb_size // self.block_size, self.block_size]
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size) 
        #[n_hti, self.emb_size // self.block_size, self.block_size, 1] * [n_hti, self.emb_size // self.block_size, 1, self.block_size]
        r = hs * ts 
        logits = self.bilinear(bl)

        return logits,r


