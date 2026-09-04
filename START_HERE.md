# Start here

This folder is the context kit, not the code. Claude Code writes the code; these files
keep it inside the lines. Delete this file once you are running.

---

## 1. Put it in place

```bash
mkdir -p ~/IdeaProjects/mtrt && cd ~/IdeaProjects/mtrt
# copy the contents of this kit into that folder, then:
git init
chmod +x scripts/rename.sh
git add -A && git commit -m "Project context: design decisions, roadmap, conventions"
```

If you want a different project name, run `scripts/rename.sh <yourname>` now, before
any code exists. It is one command now and a painful refactor later.

## 2. Install the toolchain

```bash
brew install cmake
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy onnx onnxruntime
```

GoogleTest comes in through CMake `FetchContent`, so there is nothing to install for it.
ONNX C++ headers are not needed until Week 6.

## 3. Start Claude Code

```bash
claude
```

`CLAUDE.md` loads automatically. So do the four slash commands in `.claude/commands/`:

| Command | Use |
|---|---|
| `/newop Conv2D` | Add an operator, golden test first, in the right order |
| `/bench added panel packing` | Run it properly and record real numbers in RESULTS.md |
| `/week` | Audit progress against the roadmap, honestly |
| `/explain src/memory/planner.cpp` | Interview-depth explanation, C++ idioms named |

`/week` is the one that matters most. Run it every Sunday. It checks the repo rather
than trusting the conversation, which is how you catch drift before it costs a weekend.

---

## 4. The kickoff prompt

Paste this as your first message. It asks for a plan before code on purpose.

> Read CLAUDE.md, docs/DESIGN.md and docs/ROADMAP.md first.
>
> We are starting Week 1. Before writing any code, give me an implementation plan for
> the Week 1 scope covering: the CMake layout including GoogleTest via FetchContent, the
> Tensor class design, the Graph IR types, the op registry signature, and the naive
> Executor. For each, state the design choice you are making and the alternative you
> rejected.
>
> Two things I want you to be explicit about because my C++ is rusty and I need to be
> able to defend these in an interview:
>
> 1. The exact ownership model for Tensor. When does it own storage, when is it a view,
>    what happens on copy, what happens on move, and how do we make it hard to
>    accidentally deep-copy in a hot loop.
> 2. The op registry signature. Show me the type of a kernel function and how the
>    executor dispatches, and confirm it does not force a virtual call per node.
>
> Do not write code yet. Give me the plan and wait for my approval.

## 5. How to work with it after that

A few habits that matter more than they sound:

- **Plan mode for anything structural.** Shift+Tab twice to enter it. Approve the plan,
  then let it build. Reviewing a plan is much cheaper than reviewing a diff.
- **Commit per optimization, with the measured effect in the message.** This history is
  literally your ablation study in Week 8. Do not batch commits.
- **Read the code it writes.** You are building this to be able to talk about it. Use
  `/explain` on anything you would not want to be asked about on a whiteboard. This is
  the single highest-value habit in the whole project, and the easiest one to skip.
- **When it wants to add an operator you did not ask for, say no.** Scope creep is the
  main way this project dies.

## 6. Where I can help from here

Bring me back:

- The Week 1 plan before you approve it, if you want a second opinion
- Benchmark numbers to sanity-check, especially the GEMM ladder in Week 5. I can tell
  you whether a given GFLOP/s figure is plausible for your chip or whether something is
  being measured wrong
- The README and RESULTS write-up in Week 8
- Any point where you are stuck for more than an hour
