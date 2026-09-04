---
description: Explain a piece of this codebase at interview depth
argument-hint: <file, function, or concept>
---

Explain: $ARGUMENTS

I am building this project partly to be able to discuss it fluently in systems
interviews, and my modern C++ is rusty. Explain at that level:

1. **What it does**, concretely, in the context of this codebase.

2. **The C++ mechanics.** Call out the idioms in play: ownership and lifetime, move
   versus copy, why `const`/`noexcept`/`constexpr` appear where they do, any template
   or overload resolution subtlety. Assume I know C++ but have not written it seriously
   in about three years, so name the idiom rather than silently using it.

3. **Why it is built this way.** Reference `docs/DESIGN.md` if a recorded decision
   applies. If the code contradicts a recorded decision, say so.

4. **The systems reasoning**: memory layout, cache behavior, allocation, indirection
   cost. Where relevant, roughly what this costs.

5. **The interview question this maps to**, and a two-to-three sentence answer I could
   actually say out loud. Not a script, just the shape of a good answer.

Be direct about anything in the current implementation that is weak, naive, or would not
survive scrutiny from someone who does this professionally.
