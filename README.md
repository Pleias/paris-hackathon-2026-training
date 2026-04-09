# LLM Training Hackathon

## The Challenge
Train the best language model you can in **10 minutes of GPU time** on a 32-GPU cluster.  
Judged on **validation loss** (perplexity, lower is better). *(HellaSwag accuracy as tiebreaker.)*

---

- Pre-tokenized `uint16` binary shards in `/home/data/chunk*` — vocab size **32,000**

## What You Submit
**Two Python files** (`model.py` and `train.py`) plus an optional `requirements.txt`.

- No embedded binary blobs or external assets
- `requirements.txt` packages are installed before the clock starts
- Any data preprocessing must happen within the training script
- The 10-minute GPU clock starts from the **first forward pass**

We evaluate by running your `train.py` (which must produce `checkpoint.pt`), then loading it with your `model.py`:
```python
ckpt  = torch.load("checkpoint.pt", weights_only=True)
model = get_model(ckpt["config"])   # from your model.py
model.load_state_dict(ckpt["model"])
```

`model.py` must expose `get_model(config: dict) -> nn.Module`. The `config` is whatever
dict you saved — use it to store the hyperparameters needed to reconstruct your architecture.
`forward` must be `(idx, targets=None) -> (logits, loss)`.

---

## Rules
- No external pretrained weights or training data
- Everything else goes — custom kernels, custom optimizers, ensembles, multiple training runs, whatever
- **10 minutes of GPU time**, counted from the first forward pass to last forward pass (i.e., finishing the step-inflight is fine (if this is a few seconds), staring a new one is not)
- SLURM wall time is **12 minutes** to allow for NCCL init and checkpoint saving

### yarik remark: 
- eval set is 1024 seq_len
---

## Starter Code
`model.py`, `train.py`, and `submit.sh` are provided as a working GPT baseline — DDP, bfloat16,
cosine schedule, and the 10-minute timer built in. **You are free to ignore them entirely**
and bring your own stack, as long as the checkpoint contract above is respected.
