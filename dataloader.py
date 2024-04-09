from torch.utils.data import Dataset, DataLoader
import torch
from prepro import read_docred 

class DocRED_dataset(Dataset): 
    def __init__(self, train_file, dev_file, test_file, mode, tokenizer,max_seq_length): 
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.mode = mode  
   
        if self.mode=='test':
            self.dev_features = read_docred(dev_file, tokenizer, max_seq_length=max_seq_length)                    
        else:    
            self.train_features = read_docred(train_file, tokenizer, max_seq_length=max_seq_length)           
   
    def __getitem__(self, index):
        if self.mode=='train':    
            return self.train_features[index]
                   
        elif self.mode=='test':
            return self.dev_features[index]
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_features)
        else:
            return len(self.dev_features)    


class DocRED_dataloader():  
    def __init__(self,train_file, dev_file, test_file, tokenizer, batch_size, num_workers, distributed, max_seq_length):
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed = distributed
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length


    def run(self): 

        train_dataset = DocRED_dataset(self.train_file, self.dev_file, self.test_file, 'train', self.tokenizer,self.max_seq_length) 
        test_dataset = DocRED_dataset(self.train_file, self.dev_file, self.test_file, 'test', self.tokenizer,self.max_seq_length) 
      
        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            self.train_sampler = None
            eval_sampler = None
            test_sampler = None

        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=self.train_sampler,
            drop_last=True)                                              
             
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=test_sampler)                                     
    
        return train_loader,test_loader

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    # sents = [f["sents"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask, labels, entity_pos, hts)
    return output    