# Record the README demo GIF with charmbracelet/vhs:
#
#   brew install vhs        # or: https://github.com/charmbracelet/vhs
#   vhs docs/demo.tape      # writes docs/demo.gif
#
# Requires the exported model (python python/export_gpt2_hf.py --seq-len 64) and
# `pip install torch transformers`. Then uncomment the ![demo] line in README.md.

Output docs/demo.gif

Set FontSize 20
Set Width 1100
Set Height 500
Set Padding 20
Set Theme "Dracula"

Type "python python/gpt2_generate.py --prompt 'The quick brown fox' --max-new 20"
Sleep 500ms
Enter
Sleep 8s
