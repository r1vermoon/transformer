from math import ceil, floor

with open(".data/datasets/Multi30k/train.de") as f:
    texts=f.readlines()
print(len(texts))

batch_size=5

class Data_loader:

    def __init__(self,datasets,batch_size,collate_func=None):
        self.i=-1
        self.datasets=datasets
        self.batch_size=batch_size
        self.collate_func = collate_func
        
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.i+=1
        if self.i == ceil(len(self.datasets) / batch_size) :
            raise StopIteration()
        batch = self.datasets[self.i*self.batch_size: (self.i+1)*self.batch_size]
        
        if self.collate_func != None:
            return self.collate_func(batch)
        else:
            return batch

def collate_func(batch, postfix):
    return [data + postfix for data in batch]

if __name__ == "__main__":
    post_fix = "aaa"

    def wrapper_collate_func(batch):
        return collate_func(batch, post_fix)

    data_loader = Data_loader(texts, batch_size, wrapper_collate_func)
    
    print(next(data_loader))