# VQ-VAE Animation Generator - Claude Instructions

## Project Overview
This is a PyTorch VQ-VAE project for generating looping animations. Swedish comments are INTENTIONAL and part of the author's personal style.

## Critical Constraints
⚠️ **NEVER MODIFY** without explicit approval:
- `models/vqvae_model.py` → VectorQuantizer class, EMA logic, _recover_dead_codes() method
- Any mathematical operations in the model
- Swedish text (comments, prints, plot labels) - this is the author's voice

## Code Style
- Swedish comments and logs are intentional - DO NOT translate to English
- Add type hints to all functions (using Python 3.9+ syntax)
- Add docstrings to all functions (Swedish docstrings are fine)
- Use dataclasses for configuration (see config.py)
- Use Path objects from pathlib, not strings
- Error handling: sys.exit(1), not exit()

## Workflow
- Run tests after making changes: `python -m pytest` (if tests exist)
- Check for encoding bugs: Look for mojibake (ðŸ"¦, âœ¨, FÃ¶r, TrÃ¤ning)
- Use relative imports: `from models.vqvae_model import VQVAE`

## Testing & Verification
- For model changes: Verify shapes are correct with small batch
- For config changes: Check all files import correctly
- For utility functions: Test with edge cases

## Repository Guardrails
- No changes to model architecture, training behavior, EMA updates, dead-code recovery, hyperparameters, the supplied dataset (`data/`), or committed visual evidence (`Images/`) without explicit maintainer approval
- Documentation, dependency metadata, tests, `.gitignore`, licensing, and reproducibility instructions may be updated when supported by verified repository behavior
- Do not regenerate, replace, or delete committed visual assets unless explicitly requested
- Historical metrics that were not newly reproduced must be labeled as reported results, not current benchmarks

## Repository Etiquette
- Commit messages in English for GitHub audience
- Keep Swedish in code but English in git/docs

## Device Configuration
- Primary: MPS (Apple Silicon)
- Fallback: CUDA → CPU
- Use `torch.device()` with proper fallback logic
