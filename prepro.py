from tqdm import tqdm
import ujson as json

docred_rel2id = json.load(open('/home/user/lyx/Programs/ATLOP-main/dataset/docred/meta/rel2id.json', 'r'))
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res


def read_docred(file_in, tokenizer, max_seq_length=1024,):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)
    for sample in tqdm(data, desc="Example"):
    # for x in tqdm(data, desc="Example"):
    #     for sample in x:
            sents = []     #分词后的句子
            sent_map = []  #分词前后对应关系

            entities = sample['vertexSet']
            entity_start, entity_end = [], []
            for entity in entities: #获得实体位置
                for mention in entity:
                    sent_id = mention["sent_id"] 
                    pos = mention["pos"]
                    entity_start.append((sent_id, pos[0],))
                    entity_end.append((sent_id, pos[1] - 1,))
            for i_s, sent in enumerate(sample['sents']):#分词
                new_map = {}
                for i_t, token in enumerate(sent):
                    tokens_wordpiece = tokenizer.tokenize(token)
                    if (i_s, i_t) in entity_start:
                        tokens_wordpiece = ["*"] + tokens_wordpiece
                    if (i_s, i_t) in entity_end:
                        tokens_wordpiece = tokens_wordpiece + ["*"]
                    new_map[i_t] = len(sents)
                    sents.extend(tokens_wordpiece)
                new_map[i_t + 1] = len(sents)
                sent_map.append(new_map)

            train_triple = {} #实体关系对应
            if "labels" in sample:
                for label in sample['labels']:
                    evidence = label['evidence']
                    r = int(docred_rel2id[label['r']])
                    if (label['h'], label['t']) not in train_triple:
                        train_triple[(label['h'], label['t'])] = [
                            {'relation': r, 'evidence': evidence}]
                    else:
                        train_triple[(label['h'], label['t'])].append(
                            {'relation': r, 'evidence': evidence})

            entity_pos = [] #分词后实体对应的位置
            for e in entities:
                entity_pos.append([])
                for m in e:
                    start = sent_map[m["sent_id"]][m["pos"][0]]
                    end = sent_map[m["sent_id"]][m["pos"][1]]
                    entity_pos[-1].append((start, end,))

            label_id = []
            id = []
            relations, hts = [], []  
            for h, t in train_triple.keys():#正样本
                relation = [0] * len(docred_rel2id)
                id=[]
                for mention in train_triple[h, t]:
                    relation[mention["relation"]] = 1 #正样本
                    id.append(mention["relation"])                   
                relations.append(relation)
                hts.append([h, t])
                label_id.append(id)
                pos_samples += 1

            for h in range(len(entities)):#负样本
                for t in range(len(entities)):
                    if h != t and [h, t] not in hts:
                        relation = [1] + [0] * (len(docred_rel2id) - 1) #负样本[1,0,0,...]
                        relations.append(relation)
                        hts.append([h, t])
                        label_id.append([0])
                        neg_samples += 1

            assert len(relations) == len(entities) * (len(entities) - 1)

            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            i_line += 1
            feature = {'input_ids': input_ids, #分词后句子对应的编码
                    'entity_pos': entity_pos, #实体的位置
                    'labels': relations, 
                        #关系矩阵，与hts一一对应，哪位有1表示哪个关系
                    'hts': hts, #所有实体对
                    'title': sample['title'], #标题
                    'sents': sents, #
                    'label_id': label_id,                   
                    }
            # print(sents)
            features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features
