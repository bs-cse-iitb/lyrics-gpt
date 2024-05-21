import torch
from torch.nn import functional as F

import sys
sys.path.append('../')

from src.utils import set_seed, sample

set_seed(42)

from src.dataset import CharDataset
from src.model import GPT, GPTConfig
from src.trainer import Trainer, TrainerConfig


def generate_lyrics(context = 'Tu hi yeh mujkho bata dey', output_len = 500):

    block_size = 128 # Maximum context length allowed
    train_text = open('./data/train.txt', 'r').read() 
    train_dataset = CharDataset(train_text, block_size) 

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)

    trainer_config = TrainerConfig(max_epochs=5, batch_size=128, learning_rate=6e-4, lr_decay=True, warmup_tokens=512*20,       # Trainer configuration
                               final_tokens=2*len(train_dataset)*block_size, num_workers=4, ckpt_path=None)
    trainer = Trainer(model=model, train_dataset=train_dataset, test_dataset=None, config =trainer_config)  
    
    model.load_state_dict(torch.load('./models/our_trained_model.pt',map_location=trainer.device))

    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
    y = sample(model, x, output_len, temperature=1.0, sample=True, top_k=10)[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])

    return completion