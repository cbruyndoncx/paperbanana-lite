# Reference Set Format

Reference sets provide few-shot examples for in-context learning. They are stored in a directory with an `index.json` file and associated images.

## Directory Structure

```
reference-set/
├── index.json
├── image_001.png
├── image_002.png
└── ...
```

## index.json Schema

```json
{
  "metadata": {
    "name": "Methodology Diagrams",
    "description": "Curated set of NeurIPS-style methodology diagrams"
  },
  "examples": [
    {
      "id": "unique_id",
      "source_context": "The methodology text from the paper...",
      "caption": "Figure 1: Overview of the proposed framework...",
      "image_path": "image_001.png",
      "category": "optional_category"
    }
  ]
}
```

## Fields

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique identifier for the example |
| `source_context` | Yes | Methodology/data text the diagram illustrates |
| `caption` | Yes | Figure caption or visual intent |
| `image_path` | Yes | Path to reference image (relative to index.json directory) |
| `category` | No | Optional grouping category |

## Notes

- Image paths can be relative (resolved against the index.json directory) or absolute
- The default reference set ships with the full PaperBanana repo at `data/reference_sets/` (13 methodology diagrams)
- For best results, include 10+ diverse examples covering different diagram styles
