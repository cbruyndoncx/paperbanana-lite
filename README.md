# PaperBanana Lite

Single-file academic illustration generation pipeline. Generates publication-quality methodology diagrams and statistical plots from text descriptions using a multi-agent pipeline powered by Google Gemini.

This is a standalone distillation of the full [PaperBanana](https://github.com/LLMsResearch/PaperBanana) framework (~3,000 lines across 51 files) into one executable Python script (~1,400 lines).

## Quick Start

```bash
# Install dependencies
pip install google-genai pillow tenacity

# Set your API key
export GEMINI_API_KEY="your-key"

# Generate a methodology diagram
python paperbanana_lite.py generate \
  --input methodology.txt \
  --caption "Overview of our encoder-decoder architecture"

# Generate a statistical plot
python paperbanana_lite.py plot \
  --data results.json \
  --intent "Bar chart comparing model accuracy"
```

## How It Works

Two-phase multi-agent pipeline, all running through Gemini:

**Phase 1 — Linear Planning:**
1. **Retriever** — selects relevant reference examples from a curated set via VLM
2. **Planner** — generates a detailed textual description via in-context learning
3. **Stylist** — refines the description with NeurIPS-style aesthetic guidelines

**Phase 2 — Iterative Refinement (up to N iterations):**
4. **Visualizer** — renders description into image (Gemini image gen for diagrams, matplotlib code execution for plots)
5. **Critic** — evaluates image quality, provides revision feedback; loops back to step 4

## CLI Reference

### `generate` — Methodology diagrams

```bash
python paperbanana_lite.py generate \
  --input methodology.txt \
  --caption "Figure caption" \
  --reference-dir path/to/references \
  --iterations 3 \
  --output-dir outputs
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--input` | Yes | — | Path to text file with methodology section |
| `--caption` | Yes | — | Figure caption / communicative intent |
| `--reference-dir` | No | `data/reference_sets` | Path to reference set directory |
| `--iterations` | No | 3 | Number of refinement iterations |
| `--output-dir` | No | `outputs` | Base output directory |

### `plot` — Statistical plots

```bash
python paperbanana_lite.py plot \
  --data results.json \
  --intent "Visual intent description" \
  --reference-dir path/to/references \
  --iterations 3 \
  --output-dir outputs
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--data` | Yes | — | Path to JSON file with raw data |
| `--intent` | Yes | — | Visual intent / figure caption |
| `--reference-dir` | No | `data/reference_sets` | Path to reference set directory |
| `--iterations` | No | 3 | Number of refinement iterations |
| `--output-dir` | No | `outputs` | Base output directory |

## Environment Variables

```bash
export GEMINI_API_KEY="your-key"
```

Also accepts `GOOGLE_API_KEY` as a fallback.

## Output

Results are saved to `outputs/run_<timestamp>/`:

```
outputs/run_20250215_143022_a1b2c3/
  final_output.png          # Final generated image
  planning.json             # Retrieved examples and description
  iter_1.png                # Image from iteration 1
  iter_1_details.json       # Description and critic feedback
  iter_2.png                # ...
  iter_2_details.json
```

## Reference Sets

The `--reference-dir` should point to a directory containing an `index.json` with curated reference examples. Each example has:
- `id` — unique identifier
- `source_context` — methodology text
- `caption` — figure caption
- `image_path` — path to the reference image (relative to the directory)

You can use the reference sets from the full PaperBanana repo at `data/reference_sets/`.

## Dependencies

Only 3 external packages:

- `google-genai` — Gemini VLM + image generation
- `pillow` — image I/O
- `tenacity` — retry with exponential backoff

## License

MIT
