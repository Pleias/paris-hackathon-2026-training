## Main variables
- Model architecture and size
- Training library efficiency
- Hyperparams

## Models
- Dense -> what architecture
- MoE

## Libraries 
- Torchtitan 
- Nemotron -> claude
- flash attn 4

## Hyperparams
- Extrapolation from model size and - architecture

## Other 
Engram (deepseek)


Increase to 800M
test different model sizes with standard gpt

test other architectures (with positional encoding, llama, ???)
self.fc   = nn.Linear(n_embd, 4 * n_embd, bias=False) -> how for llama