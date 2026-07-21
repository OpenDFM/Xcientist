OpenHarness vendored source
===========================

Source: https://github.com/HKUDS/OpenHarness
Commit: 9b2efd795c6aa09f88b0c257d269a9e518da6ae7
Imported for: Xcientist Experiment Agent runtime

This copy is intentionally vendored instead of imported from the user's
global `oh` installation. Experiment Agent must isolate its OpenHarness
configuration, data, logs, permissions, hooks, and memory under each
experiment workspace.

Local modifications should be made in this tree when Xcientist needs
runtime-level behavior that upstream OpenHarness does not expose cleanly.
