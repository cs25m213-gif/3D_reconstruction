# Source publication checks

Validation performed on 17 November 2026 using the existing workstation Python environment, with imports resolved from this repository checkout.

- Parsed every exported Python file and JSON configuration.
- Checked shell syntax of the root launchers and module wrapper.
- Ran 28 tests from `test_available_selector_data`, `test_reconstruction_io` and `test_selector_robustness`: all passed.
- Opened reconstruction and complete-selector CLI help from this checkout.
- Checked staged file types, sizes and paths: no runs directory, datasets, weights, meshes, archives, virtual environments or private key files included.
- Scanned candidate text for common private-key, GitHub-token, Hugging Face-token and AWS access-key patterns; no matches found.
- Retained third-party licenses for the included Three.js and OpenMVS helper code.

This publication did not execute full training, native GPU reconstruction or optional third-party builds. Those require separately provisioned data, checkpoints and runtime dependencies.
