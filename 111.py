from tqdm.auto import tqdm
import time
import torch

if __name__ == "__main__":
    train_epoch = 100
    a = [torch.randn(3, 4) for i in range(10)]
    print(len(a))
    #from list to tensor
    b = torch.stack(a)
    print(b.shape)
    

