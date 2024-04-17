import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

class MoPro(nn.Module):

    def __init__(self, base_encoder, args, config, model):
        super(MoPro, self).__init__()
        self.nums_labels = 1
        self.low_dim = args.low_dim
        self.num_class = args.num_class
        
        #encoder
        self.encoder_q = base_encoder(config, model, num_class=args.num_class)
        #momentum encoder
        self.encoder_k = base_encoder(config, model, num_class=args.num_class)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


        # 投影器
        if args.low_dim != -1:
            self.projection =  nn.Sequential(
                nn.Linear(config.hidden_size, self.low_dim),
                Normalize(2)
                )
            self.projection_back = nn.Linear(self.low_dim,config.hidden_size)
        #
        self.relation =  nn.Linear(self.low_dim*2,1)
        #
        self.classifier_projection = nn.Linear(self.low_dim,self.num_class)

        # 设置原型
        self.register_buffer("prototypes", torch.zeros(args.num_class, self.low_dim))
        self.register_buffer("prototypes_visited", torch.zeros(args.num_class))
        self.register_buffer("prototypes_density", torch.ones(args.num_class)*args.temperature)
        self.register_buffer("prototypes_distance", torch.zeros(args.num_class))

        #loss
        self.loss_ce = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, args):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size < args.moco_queue :
            self.queue[:, ptr:ptr + batch_size] = keys.T
        elif ptr + batch_size < 2 *args.moco_queue:
            k1 = keys[0:args.moco_queue-ptr-1]
            k2 = keys[args.moco_queue-ptr-1:-1]
            self.queue[:, ptr:-1] = k1.T
            self.queue[:, 0:ptr + batch_size-args.moco_queue] = k2.T
        else:
            k1 = keys[0:args.moco_queue-ptr-1]
            k2 = keys[args.moco_queue-ptr-1:2*args.moco_queue-ptr-2]
            self.queue[:, ptr:-1] = k1.T
            self.queue[:, 0:args.moco_queue-1] = k2.T
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        # batch_size_this = x.shape[0]
        # x_gather = concat_all_gather(x)
        # batch_size_all = x_gather.shape[0]
        batch_size_this = x[0].shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather[0].shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        # batch_size_this = x.shape[0]
        # x_gather = concat_all_gather(x)
        # batch_size_all = x_gather.shape[0]
        batch_size_this = x[0].shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather[0].shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    

    def forward(self, batch, args, \
                is_eval=False, is_proto=False, is_clean=False,\
                    is_proto_init=0,is_analysis=False):

        # 原型初始化
        if is_proto_init == 1:
            # 初始化为0
            self._zero_prototype_features()
            return
        if is_proto_init == 3:
            # 原型特征平均
            self._initialize_prototype_features()
            return
        if is_proto_init == 4:
            ## 更新分类输出的各类temperature
            self._update_prototype_density()
            return
        
        # 输入：文本、mask、标签、实体位置、实体对、类别标签
        input_ids = batch[0].cuda(args.gpu, non_blocking=True)        
        attention_mask = batch[1].cuda(args.gpu, non_blocking=True) 
        labels = batch[2]
        entity_pos = batch[3]
        hts = batch[4]
        label_id = batch[5].cuda(args.gpu, non_blocking=True)
        target = label_id
        

        if is_proto_init == 2:
            ## 原型特征初始化，按类累加
            with torch.no_grad():  # no gradient
                if not is_eval:
                    # shuffle for making use of BN
                    # img, idx_unshuffle = self._batch_shuffle_ddp(img)
                    _, k = self.encoder_k(input_ids,attention_mask,labels,entity_pos,hts) 
                    k_compress = self.projection(k)
                    features = k_compress
                    targets = label_id
                    # undo shuffle
                    # k_compress = self._batch_unshuffle_ddp(k_compress, idx_unshuffle)
                    # # gather all features across gpus
                    # features = concat_all_gather(k_compress)
                    # # gather all targets across gpus
                    # targets = concat_all_gather(target)
                else:
                    _, k = self.encoder_k(input_ids,attention_mask,labels,entity_pos,hts) 
                    features = self.projection(k)
                    targets = label_id
                for feat, label in zip(features, targets):
                    for l in label:
                        if l != 0:
                            self.prototypes[int(l)] += feat
                            self.prototypes_visited[int(l)] += 1  # count visited times for average
            return        
        
        
        # output:分类结果，q:实体对特征
        output,q = self.encoder_q(input_ids,attention_mask,labels,entity_pos,hts)
        q_compress = self.projection(q)

        # 特征压缩与重建
        if is_proto:
            ## 更新重建部分仅仅依赖于q而不需要反传梯度至特征提取器
            q_reconstruct = self.projection_back(q_compress.detach().clone())
        else:
            ## 同时更新压缩和重建部分
            q_reconstruct = self.projection_back(self.projection(q.detach().clone()))

        # 测试
        if is_eval:
            output = get_label(output,num_labels=self.nums_labels)     
            return output, q_reconstruct, q


        #区分数据来源
        if is_eval:
            domain = batch[2].cuda(args.gpu, non_blocking=True)
        else:
            domain = batch[3].cuda(args.gpu, non_blocking=True)
        fewshot_idx = (domain > 0).view(-1)


        # 动量更新
        with torch.no_grad():  # no gradient 
            self._momentum_update_key_encoder(args)  # update the momentum encoder
        
        # 基于prototype的对比学习
        if is_proto:
            prototypes = self.prototypes.detach().clone()
            logits_proto = torch.mm(q_compress, prototypes.t())
            logits_proto_raw = logits_proto.detach().clone()
            ## 针对fewshot样本 可以考虑margin来尤其拉近距离
            if args.margin != 0:
                ## Additive margin softmax
                # target_fewshot = target[fewshot_idx]
                # logits_proto[fewshot_idx,target_fewshot] -= args.margin 
             if args.use_temperature_density:
                ## 可以根据每个类别的密度紧致程度进行缩放
                density_temperature = self.prototypes_density.detach().clone().view(1, -1)
                logits_proto = logits_proto/density_temperature
                logits_proto_raw = logits_proto_raw/density_temperature
            else:
                logits_proto = logits_proto/args.temperature
                logits_proto_raw = logits_proto_raw/args.temperature
            with torch.no_grad():
                ## 每个样本输出是由High-Dim分类器输出概率&样本-prototype相似度加权得到的
                target_soft = args.alpha*F.softmax(output, dim=1) + (1-args.alpha)*F.softmax(logits_proto_raw, dim=1)
                ## 注意生成target soft时保证fewshot样本的标签不会改变
                target_soft[fewshot_idx] = F.one_hot(target[fewshot_idx].long(), num_classes=args.num_class).float()           
        else:
            logits_proto = 0
            # target_soft = F.one_hot(target, num_classes=args.num_class)
            target_soft = F.softmax(output, dim=1)

        #噪声过滤
        gt_score = target_soft[target>=0,target]
        correct_idx = fewshot_idx | (gt_score>1./args.num_class)
        if is_proto and is_clean:
            if is_clean == 1:
                clean_idx_pred = gt_score>(1/args.num_class)
                ## 分配Pred标签的前提是该标签的值大于阈值, 可以覆盖GT标签
                max_score, hard_label = target_soft.max(1)
                correct_idx = max_score>args.pseudo_th
                ## 从correct_idx中剔除fewshot index样本(不参与标签修改)
                correct_idx = correct_idx & (~fewshot_idx)
                ## 伪标签=>修改图像标签为预测置信度高的类别
                target[correct_idx] = hard_label[correct_idx]
                clean_idx = clean_idx_pred | correct_idx | fewshot_idx
            if (not is_eval) and (not is_analysis):
                #置信度较高的样本更新原型
                with torch.no_grad():  # no gradient
                    # shuffle & undo shuffle for making use of BN
                    # img, idx_unshuffle = self._batch_shuffle_ddp(img)
                    # k_compress = self.projection(self.encoder_k(img))
                    # k_compress = self._batch_unshuffle_ddp(k_compress, idx_unshuffle)
                    # features = concat_all_gather(k_compress)
                    _, k = self.encoder_k(input_ids,attention_mask,labels,entity_pos,hts) 
                    k_compress = self.projection(k)
                    features = k_compress
                    ## 记录下所有样本距离对应(伪)类别prototype的距离l2
                    prototypes = self.prototypes.clone().detach()
                    # targets = concat_all_gather(target).view(-1)
                    targets = target
                    prototype_targets = torch.index_select(prototypes, dim=0, index=targets.view(-1).type(torch.int64))
                    dists_prototypes = torch.norm(features-prototype_targets, dim=1)
                    ## 按照FoPro的方法是仅仅保留clean样本来更新prototype
                    # clean_idx_all = concat_all_gather(clean_idx.long()) 
                    clean_idx_all = clean_idx_all.bool()
                    # update momentum prototypes with pseudo-labels
                    for feat, label, dist in zip(features[clean_idx_all],\
                        targets[clean_idx_all], dists_prototypes[clean_idx_all]):
                        self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat
                        self.prototypes_visited[label] += 1  # 记录更新当前类prototype的样本数量
                        self.prototypes_distance[label] += dist  # 记录下该样本距离类prototype的L2距离
                    ## normalize prototypes
                    self.prototypes = F.normalize(self.prototypes, p=2, dim=1)


        with torch.no_grad(): # no gradient
            if is_proto:
                ## 1) fewshot样本
                ## 2) 与各个prototype之间的相似度 & prototype与prototype之间的相似度的比值
                prototypes = self.prototypes.detach().clone()
                proto_proto = torch.mm(prototypes, prototypes.t())  ## N_cls * N_cls
                proto_proto_sim = torch.index_select(proto_proto, dim=0, index=target.view(-1))
                logits_proto_sim = torch.mm(q_compress, prototypes.t()) ## N_batch * N_cls
                ## 度量方式a. 计算二者分布差的L2范数=>最大值归一化
                # dist_sim = torch.norm(proto_proto_sim - logits_proto_sim, dim=1)/(math.sqrt(2**2 * args.num_class))
                dist_sim_diff = torch.abs(proto_proto_sim - logits_proto_sim)  ## N_batch * N_cls
                dist_sim = 0.5 * dist_sim_diff[target>=0, target]  ## 点选每个样本与对应类别的prototype距离
                dist_sim_diff[target>=0, target] = 0  ## 抹掉该距离方便后续加权
                dist_sim += 0.5 * torch.sum(dist_sim_diff, dim=1)
                if (not is_clean):
                    arcface_idx = fewshot_idx | (dist_sim <= args.dist_th)
                else:
                    ## 使用clean的准则挑选样本
                    arcface_idx = clean_idx
            else:
                ## 没有fewshot样本只能用所有高置信度样本
                arcface_idx = fewshot_idx | (gt_score>args.pseudo_th)


        img_aug = batch[2].cuda(args.gpu, non_blocking=True)
        q_aug_compress = self.projection(self.encoder_q(img_aug))
        if torch.sum(arcface_idx) > 0:
            q_compress_arcface = q_compress[arcface_idx]
            q_aug_compress_arcface = q_aug_compress[arcface_idx]
            output_arcface = torch.cat((self.classifier_projection(q_compress_arcface),\
                self.classifier_projection(q_aug_compress_arcface)), dim=0)
            target_arcface = torch.cat((target[arcface_idx], target[arcface_idx]), dim=0)
            ## additive margin & temperature softmax (输出0~1之间)
            if args.margin != 0:
                output_arcface[target_arcface>=0,target_arcface] -= args.margin
            output_arcface /= args.temperature
            # print("size output arcface {} target arcface {}".format(output_arcface.size(), target_arcface.size()))
        else:
            output_arcface, target_arcface = None, None


        
        if args.pre_relation or args.ft_relation:
            ## 一开始训练 & 微调 relation时使用arcface idx
            relation_idx = arcface_idx
        else:
            ## 后续只要是干净样本都加入训练
            relation_idx = correct_idx
        N_random = 4
        if is_relation and is_proto and torch.sum(relation_idx) > 0:
            ## 必须先得有prototype才能更新训练只利用置信度很高的样本来进行更新
            ## 首先构建正负样本对
            with torch.no_grad():
                prototypes = self.prototypes.detach().clone()
                q_compress_relation = q_compress[relation_idx].detach().clone()
                # q_aug_compress_relation = q_aug_compress[relation_idx].detach().clone()
                N_relation = q_compress_relation.size(0)
                target_relation = target[relation_idx]
                q_compress_relation_tiled = torch.repeat_interleave(q_compress_relation, repeats=args.num_class, dim=0)
                prototypes_relation_tiled = prototypes.repeat(N_relation, 1)
            relation_score = self.relation(torch.cat((q_compress_relation_tiled, prototypes_relation_tiled), dim=1))
            relation_target = target_relation
            relation_score = relation_score.view(N_relation, args.num_class)
        else:
            relation_score, relation_target = None, None
        ##=========================================================================##
        ## 仅仅更新干净样本
        ##=========================================================================##
        if is_clean and is_proto:
            output = output[clean_idx]
            target = target[clean_idx]
            logits_proto = logits_proto[clean_idx]
            target_soft = target_soft[clean_idx]
            gt_score = gt_score[clean_idx]

        #loss
        labels = [torch.tensor(label) for label in labels]
        labels = torch.cat(labels, dim=0).to(logits)
        loss_proto = 0
        if is_proto:
            loss_proto = args.w_proto * self.loss_ce(logits_proto, labels) 
        
        loss_cls = self.loss_ce(output, labels)   
        # instance contrastive loss
        loss_inst = self.loss_ce(logits, inst_labels)  
        
        loss = loss_cls + args.w_inst*loss_inst + loss_proto 

        output = get_label(output,num_labels=self.nums_labels)            

        return output,loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def get_label(logits, num_labels=-1):
    th_logit = logits[:, 0].unsqueeze(1)
    output = torch.zeros_like(logits).to(logits)
    mask = (logits > th_logit)
    if num_labels > 0:
        top_v, _ = torch.topk(logits, num_labels, dim=1)
        top_v = top_v[:, -1]
        mask = (logits >= top_v.unsqueeze(1))
    output[mask] = 1.0
    output[:, 0] = (output.sum(1) == 0.).to(logits)
    return output



class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out