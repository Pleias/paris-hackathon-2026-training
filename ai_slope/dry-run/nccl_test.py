import torch
import torch.distributed as dist
import os
import time

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

if rank == 0:
    print(f"World size: {world_size}, Backend: {dist.get_backend()}", flush=True)

# All-reduce correctness test
t = torch.ones(1024, 1024, device="cuda") * rank
dist.all_reduce(t)
expected = sum(range(world_size)) * 1024 * 1024
actual = t.sum().item()
if rank == 0:
    print(f"all_reduce: expected={expected:.0f}, got={actual:.0f}, match={abs(expected - actual) < 1}", flush=True)

# Bandwidth test (1 GiB tensor, 5 all-reduces)
size = 256 * 1024 * 1024  # 256M floats = 1 GiB
data = torch.randn(size, device="cuda")
dist.barrier()
start = time.time()
for _ in range(5):
    dist.all_reduce(data)
torch.cuda.synchronize()
elapsed = time.time() - start
if rank == 0:
    bw = 5 * 4 * size / elapsed / 1e9
    print(f"Bandwidth: {bw:.1f} GB/s over 5 all-reduces of 1 GiB", flush=True)

dist.destroy_process_group()
