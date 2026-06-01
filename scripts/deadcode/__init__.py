"""Evidence-driven dead-code elimination toolchain.

Standalone tooling (outside the `card_capture` package) that finds dead code
with vulture, scores it into a manifest, validates removals against a
tests+video+metric gate, and applies/bisects changes. See
docs/superpowers/specs/2026-05-31-dead-code-elimination-design.md.
"""
