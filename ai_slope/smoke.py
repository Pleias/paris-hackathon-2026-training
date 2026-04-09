import sys

import torch

sys.path.insert(0, "ai_slope")
from model import get_model


def main() -> None:
    cfg = {
        "hidden_size": 256,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 512,
        "vocab_size": 1024,
        "seq_len": 64,
    }

    model = get_model(cfg)
    x = torch.randint(0, cfg["vocab_size"], (2, cfg["seq_len"]))
    logits, loss = model(x, targets=x)

    print(
        f"OK: logits={logits.shape}, loss={loss.item():.4f}, "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    ckpt_path = "/tmp/test_ckpt.pt"
    checkpoint = {"step": 0, "model": model.state_dict(), "config": cfg}
    torch.save(checkpoint, ckpt_path)

    loaded = torch.load(ckpt_path, weights_only=True)
    model2 = get_model(loaded["config"])
    model2.load_state_dict(loaded["model"])
    print("Checkpoint round-trip: OK")


if __name__ == "__main__":
    main()
