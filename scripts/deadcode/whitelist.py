"""Vulture whitelist: names reached dynamically that vulture cannot see.

Vulture reports these as unused because the references are dynamic (click
decorators, pytest fixtures, dataclass fields consumed by asdict/JSON,
console-script entry points). Listing a name here marks it 'used'. Keep this
file MINIMAL — only add entries proven to be false positives, never to silence
a genuinely dead symbol.
"""
# pytest hooks / fixtures discovered by name
_.pytestmark
_.fixture

# dataclass fields serialized via dataclasses.asdict / JSON (not referenced
# by attribute access in source)
_.metadata
_.config_preset

# click command callbacks invoked by the CLI framework
_.run
