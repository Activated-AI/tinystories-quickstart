import os
import math
import time
import inspect
from dataclasses import dataclass, fields
import torch
import torch.nn as nn
from torch.nn import functional as F
import tokenizers

@dataclass
class GPTConfig:
    block_size: int = 256 # max sequence length     
    n_layer: int = 12 # number of layers    
    n_head: int = 16 # number of heads       
    n_embd: int = 512 # embedding dimension 
    vocab_size: int = 8192
    batch_size: int = 128    
    data_dir: str = 'tiny-stories-8k-eos'    
    expt_name: str = '12l_16h_512d_8k_vocab_2e3_lr_good_max_eos'
    max_lr: float = 2e-3
    min_lr: float = 2e-4
    beta_1: float = 0.9
    beta_2: float = 0.99
    warmup_steps:int = 200
    max_steps: int = 8000
    weight_decay: float = 0.13
    feed_forward_factor: int = 2  # how much bigger the MLP blocks are than the model n_embd
    # Do various hacky things (don't use torch.compile, don't load training data) to speed up the run.  
    # # We are checking for runnability rather than model quality.
    smoke_test: bool = False 

    def __str__(self):
        return '\n'.join([f'{field.name}: {str(getattr(self, field.name))}' for field in fields(self)])
        

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.feed_forward_factor * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(config.feed_forward_factor * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type, beta1, beta2):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"    
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), eps=1e-8, fused=use_fused)
        return optimizer


def load_tokens(filename):    
    return torch.load(filename).to(dtype=torch.long)
    

class DataLoaderLite:
    def __init__(self, data_dir, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}
        
        shards = os.listdir(data_dir)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_dir, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
    
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T 
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T 
        return x, y


def generate(model, enc, prompt, max_length, num_return_sequences):
    model.eval() 
    
    eos_id = enc.token_to_id('[EOS]')
    tokens = enc.encode(prompt).ids
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to('cuda')
    sample_rng = torch.Generator(device='cuda')
    sample_rng.manual_seed(42)
    while xgen.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (B, 50), topk_indices is (B, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            xgen = torch.cat((xgen, xcol), dim=1)
    # print the generated text
    for i in range(num_return_sequences):
        # look for EOS here to truncate.
        first_eos = (xgen[i] == eos_id).nonzero()
        if first_eos.size(0) > 0:            
            this_end = first_eos[0].item()            
        else:            
            this_end = max_length
        tokens = xgen[i, :this_end].tolist()
        decoded = enc.decode(tokens)
        print(f"sample {i}: {decoded}")
    model.train()


def main():
    config = GPTConfig()
    assert torch.cuda.is_available()
    device = "cuda"
    device_type = "cuda"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tokenizers.ByteLevelBPETokenizer(f'{config.data_dir}/tokenizer-vocab.json', f'{config.data_dir}/tokenizer-merges.txt')

    val_loader = DataLoaderLite(data_dir=config.data_dir, B=config.batch_size, T=config.block_size, split="val")
    bytes_in_val_text = 19190318 
    bytes_per_token = bytes_in_val_text / val_loader.tokens.shape[0]
    if not config.smoke_test:
        train_loader = DataLoaderLite(data_dir=config.data_dir, B=config.batch_size, T=config.block_size, split="train")        
    else:    
        train_loader = val_loader

    model = GPT(config)

    torch.set_float32_matmul_precision('medium')

    model.to(device)
    use_compile = not config.smoke_test
    if use_compile:
        print('using torch.compile')
        model = torch.compile(model)
            
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < config.warmup_steps:
            return config.max_lr * (it+1) / config.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > config.max_steps:
            return config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return config.min_lr + coeff * (config.max_lr - config.min_lr)

    optimizer = model.configure_optimizers(weight_decay=config.weight_decay, learning_rate=config.max_lr, 
                                           device_type=device_type, beta1=config.beta_1, beta2=config.beta_2)

    # create the log directory we will write checkpoints to and log to
    log_dir = f'logs/{config.expt_name}'
    if config.smoke_test:
        log_dir += '_smoke'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'log.txt')
    with open(log_file, "w") as f: # open for writing to clear the file
        f.write(str(config) + "\n")

    t_start = time.time()
    last_step = False
    loss_accum = []

    def log_msg(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(f"{msg}\n")

    for step in range(config.max_steps):
        t0 = time.time()
        last_step = (step == config.max_steps - 1) or last_step

        # once in a while evaluate our validation loss
        if (step % 250 == 0 and step > 0) or last_step:
            if config.smoke_test:
                print('exiting due to smoke test')
                last_step = True
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                if last_step:
                    val_loader.reset() # start from beginning for final eval
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()

            val_loss = val_loss_accum.item()
            per_byte_loss = val_loss / bytes_per_token
            
            log_msg(f'step {step} | val loss {val_loss:.4f} | byte loss {per_byte_loss:.4f} | ds {time.time() - t_start:.1f}s')            
                
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

            if last_step:
                generate(model, enc, "Lily and her mom went to a park", 256, 4)
                break
        
        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)            
        loss = loss 
        loss_accum.append(loss.detach())
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()        

        if step % 10 == 0:
            t1 = time.time()
            dt = t1 - t0 # time difference in seconds
            ds = t1 - t_start
            tokens_processed = train_loader.B * train_loader.T
            tokens_per_sec = tokens_processed / dt
            avg_loss = sum(loss_accum) / len(loss_accum)
            loss_accum.clear()

            if ds > 600:
                print('exiting due to time limit')
                last_step = True                

            per_byte_loss = avg_loss / bytes_per_token
            log_msg(f'step {step:5d} | loss {avg_loss:.6f} | byte loss {per_byte_loss:.4f} | lr {lr:.4e} | norm {norm:.4f} | dt {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | ds {ds:.1f}s')
            

if __name__ == "__main__":
    main()