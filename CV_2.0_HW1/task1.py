import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    dataloader = DataLoader(prepare_data(), batch_size=256, shuffle=True)

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).cuda().train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    losses_history = []
    forward_times = []
    backward_times = []

    for batch_idx, (data, target) in enumerate(dataloader):
        # noise = torch.randn(data.shape).to('cuda')
        # data = data.to('cuda') + noise            
        '''
        Операции выполняются последовательно => пока GPU создаёт шум, CPU простаивает
        (level 2)
        '''   
        noise = torch.randn(data.shape)  # сначала создаём шум на CPU
        data = data.to('cuda', non_blocking=True)  # асинхронный перенос
        noise = noise.to('cuda', non_blocking=True)
        data = data + noise
        target = target.to('cuda')


        optimizer.zero_grad()

        '''
        time.time() - измеряет, когда CPU отправил операцию, а не когда GPU закончил 
        (level 3)
        '''
        starter.record()
        output = model(data)
        loss = criterion(output, target)
        ender.record()
        torch.cuda.synchronize()
        forward_times.append(starter.elapsed_time(ender) / 1000.0)

        # измеряем реальное время backward pass на GPU
        starter.record()
        loss.backward()
        ender.record()
        torch.cuda.synchronize()
        backward_times.append(starter.elapsed_time(ender) / 1000.0)

        optimizer.step()

        #losses_history.append(loss) - плохо, добавляет тензор (level 1)
        losses_history.append(loss.item())  # извлекаем число, чтобы предотвратить утечку GPU
        print(f"Batch {batch_idx} loss: {loss.item():.4f}")
        # torch.cuda.empty_cache() - вызывает синхронизацию CPU и GPU, фрагментацию памяти (level 1)

    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times)}, "
          f"avg backward time is {statistics.mean(backward_times)}")

if __name__ == '__main__':
    train()
