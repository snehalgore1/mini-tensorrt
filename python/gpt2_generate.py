"""Generate text with real GPT-2 running on MiniTensorRT, and check it matches
HuggingFace greedy decoding.

Pipeline: tokenize the prompt here (HF GPT2Tokenizer) -> pass ids to the C++
`run_gpt2` binary (the runtime does the generation loop) -> detokenize its output
here. This keeps only tokenizer I/O in Python; the inference is 100% our runtime.

Run:  python python/gpt2_generate.py --prompt "The quick brown fox" --max-new 20
      python python/gpt2_generate.py --prompt "..." --check   # compare vs HF greedy
"""

import argparse
import os
import subprocess

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUN_GPT2 = os.path.join(ROOT, "build", "tools", "run_gpt2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The quick brown fox")
    ap.add_argument("--max-new", type=int, default=20)
    ap.add_argument("--check", action="store_true", help="compare vs HF greedy")
    args = ap.parse_args()

    from transformers import GPT2Tokenizer

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    prompt_ids = tok.encode(args.prompt)

    if not os.path.exists(RUN_GPT2):
        raise SystemExit(f"missing {RUN_GPT2}; build first: cmake --build build")

    out = subprocess.run(
        [RUN_GPT2, "--ids", " ".join(map(str, prompt_ids)),
         "--max-new", str(args.max_new)],
        capture_output=True, text=True, check=True,
    )
    gen_ids = [int(x) for x in out.stdout.split()]
    text = tok.decode(prompt_ids + gen_ids)
    print("=== MiniTensorRT GPT-2 ===")
    print(text)

    if args.check:
        import torch
        from transformers import GPT2LMHeadModel

        model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        with torch.no_grad():
            hf = model.generate(
                torch.tensor(prompt_ids).unsqueeze(0),
                max_new_tokens=args.max_new, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )[0].tolist()
        hf_gen = hf[len(prompt_ids):]
        match = hf_gen[:len(gen_ids)] == gen_ids[:len(hf_gen)]
        print("\n=== HuggingFace greedy ===")
        print(tok.decode(hf))
        print(f"\n[CHECK] our gen ids == HF gen ids: {match}")
        if not match:
            print(f"  ours: {gen_ids}\n  hf:   {hf_gen}")
        raise SystemExit(0 if match else 1)


if __name__ == "__main__":
    main()
