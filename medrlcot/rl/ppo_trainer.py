"""
Stage-2 PPO with tqdm progress bars + reward print-outs.
"""

import os, torch, wandb
from tqdm import tqdm
from datasets   import load_from_disk
from trl        import PPOTrainer, PPOConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .prompts   import GEN_PROMPT
from .reward_llama3 import LlamaJudge

MAX_IN, MAX_NEW = 1_000, 1_000
MAX_SEQ         = MAX_IN + MAX_NEW

def load_data(path):
    ds = load_from_disk(path)
    return ds.remove_columns([c for c in ds.column_names if c not in ("note","target")])

def main(
    sft_ckpt="checkpoints/sft_flan",
    ppo_data="data/ppo_dataset",
    w_acc=0.4, w_cot=0.6,
    out_dir="checkpoints/ppo_flan",
    epochs=2,
):
    # ─── policy ────────────────────────────────────────────────────────────────
    policy = AutoModelForSeq2SeqLM.from_pretrained(
        sft_ckpt, load_in_4bit=True, device_map="auto")
    policy.gradient_checkpointing_enable()

    tok = AutoTokenizer.from_pretrained(sft_ckpt)
    tok.model_max_length = MAX_SEQ

    judge  = LlamaJudge()
    cfg    = PPOConfig(batch_size=1, ppo_epochs=4, learning_rate=1e-5,
                       target_kl=0.1, log_with="none")
    trainer = PPOTrainer(cfg, policy, tok)

    data = load_data(ppo_data)

    # ─── training loop with tqdm ──────────────────────────────────────────────
    for ep in range(epochs):
        pbar = tqdm(data, desc=f"Epoch {ep+1}/{epochs}", unit="sample")
        for sample in pbar:
            prompt = GEN_PROMPT.format(note=sample["note"])
            q_ids  = tok(prompt, return_tensors="pt").to(policy.device)

            r_ids = policy.generate(
                **q_ids, max_new_tokens=MAX_NEW,
                do_sample=True, temperature=0.7,
                eos_token_id=tok.eos_token_id,
            )
            pred = tok.decode(r_ids[0], skip_special_tokens=True)

            acc, cot   = judge.score(sample["note"], pred, sample["target"])
            reward_val = float(w_acc*acc + w_cot*cot)

            # PPO step (lists)
            trainer.step([prompt], [pred], [reward_val])

            pbar.set_postfix({"R": reward_val, "Acc": acc, "CoT": cot})

        ckpt_dir = os.path.join(out_dir, f"epoch_{ep}")
        trainer.save_pretrained(ckpt_dir)
        print(f"✓ saved {ckpt_dir}")

if __name__ == "__main__":
    main()
