from tqdm import tqdm
from time import time
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
import pandas as pd
from model import *
from reader import *
from sklearn.metrics import mean_squared_error
import utils


if __name__ == '__main__':
    
    # initial args
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument('--model', type=str, default='YambdaUserResponse', help='User response model class.')
    init_parser.add_argument('--reader', type=str, default='RL4RSDataReader', help='Data reader class')
    initial_args, _ = init_parser.parse_known_args()
    print(initial_args)
    modelClass = eval('{0}.{0}'.format(initial_args.model))
    readerClass = eval('{0}.{0}'.format(initial_args.reader))
    
    # control args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=9, help='random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=128, help='number of epoch')

    # customized args
    parser = modelClass.parse_model_args(parser)
    parser = readerClass.parse_data_args(parser)
    args, _ = parser.parse_known_args()
    print(args)
    
    utils.set_random_seed(args.seed)
    reader = readerClass(args)
    
    # 动态采样：最多20万条，如果数据不足则使用全部数据
    train_size = min(200000, len(reader.data['train']))
    val_size = min(200000, len(reader.data['val']))
    reader.data['train'] = reader.data['train'].sample(n=train_size, random_state=42)
    reader.data['val'] = reader.data['val'].sample(n=val_size, random_state=42)
    print(f"Using {train_size} train samples, {val_size} val samples")
    print(reader.get_statistics())
    
    device = 'cuda:0'
    model = modelClass(args, reader, device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.optimizer = optimizer
    model.load_from_checkpoint(args.model_path)
    
    print("Loading model from checkpoint done.")
    
    # evaluate on validation set
    t2 = time()
    print("evaluating on validation set")
    reader.set_phase("val")
    eval_loader = DataLoader(reader, batch_size = args.batch_size,
                             shuffle = False, pin_memory = False, 
                             num_workers = reader.n_worker)
    valid_preds, valid_true = [], []
    pbar = tqdm(total = len(eval_loader.dataset))
    with torch.no_grad():
        for i, batch_data in enumerate(eval_loader):
            # predict
            wrapped_batch = utils.wrap_batch(batch_data, device = device)
            out_dict = model.forward(wrapped_batch)
            valid_preds.append(out_dict['preds'].cpu().numpy())
            valid_true.append(batch_data['feedback'].cpu().numpy())
            pbar.update(args.batch_size)
    pbar.close()
    
    # MSE for regression task
    mse = mean_squared_error(np.concatenate(valid_true), np.concatenate(valid_preds))
    print(f"Validation MSE: {mse:.4f}")
    print(f"Validation time: {time() - t2:.4f}")
