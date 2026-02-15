#!/usr/bin/env -S uv run --quiet --script

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "google-genai",
#   "pillow",
#   "tenacity",
# ]
# ///

"""
PaperBanana Lite: Multi-agent academic illustration generation.

Usage:
    uv run paperbanana_lite.py generate --input methodology.txt --caption "Overview"
    uv run paperbanana_lite.py plot --data results.json --intent "Bar chart"
"""

import argparse
import base64
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

VLM_MODEL = "gemini-2.0-flash"
IMAGE_MODEL = "gemini-3-pro-image-preview"
NUM_RETRIEVAL_EXAMPLES = 10

# Module-level client — lazily initialized
_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key or not api_key.strip():
            print("Error: GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is required.", file=sys.stderr)
            sys.exit(1)
        _client = genai.Client(api_key=api_key)
    return _client


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

DIAGRAM_RETRIEVER_PROMPT = """\
# Background & Goal
We are building an **AI system to automatically generate method diagrams for academic papers**. Given a paper's methodology section and a figure caption, the system needs to create a high-quality illustrative diagram that visualizes the described method.

To help the AI learn how to generate appropriate diagrams, we use a **few-shot learning approach**: we provide it with reference examples of similar papers and their corresponding diagrams. The AI will learn from these examples to understand what kind of diagram to create for the target paper.

# Your Task
**You are the Retrieval Agent.** Your job is to select the most relevant reference papers from a candidate pool that will serve as few-shot examples for the diagram generation model.

You will receive:

- **Target Input:** The methodology section and caption of the paper for which we need to generate a diagram
- **Candidate Pool:** Existing papers (each with methodology and caption)

You must select the **Top {num_examples} candidates** that would be most helpful as examples for teaching the AI how to draw the target diagram.

# Selection Logic (Topic + Intent)

Your goal is to find examples that match the Target in both **Domain** and **Diagram Type**.

**1. Match Research Topic (Use Methodology & Caption):**
* What is the domain? (e.g., Agent & Reasoning, Vision & Perception, Generative & Learning, Science & Applications).
* Select candidates that belong to the **same research domain**.
* *Why?* Similar domains share similar terminology (e.g., "Actor-Critic" in RL).

**2. Match Visual Intent (Use Caption & Keywords):**
* What type of diagram is implied? (e.g., "Framework", "Pipeline", "Detailed Module", "Performance Chart").
* Select candidates with **similar visual structures**.
* *Why?* A "Framework" diagram example is useless for drawing a "Performance Bar Chart", even if they are in the same domain.

**Ranking Priority:**

1. **Best Match:** Same Topic AND Same Visual Intent (e.g., Target is "Agent Framework" -> Candidate is "Agent Framework", Target is "Dataset Construction Pipeline" -> Candidate is "Dataset Construction Pipeline").
2. **Second Best:** Same Visual Intent (e.g., Target is "Agent Framework" -> Candidate is "Vision Framework"). *Structure is more important than Topic for drawing.*
3. **Avoid:** Different Visual Intent (e.g., Target is "Pipeline" -> Candidate is "Bar Chart").

# Input Data

## Target Input

- **Caption:** {caption}
- **Methodology section:** {source_context}

## Candidate Pool
{candidates}

# Output Format
Provide your output strictly in the following JSON format, containing only the **exact Paper IDs** of the Top {num_examples} selected papers:
{{
    "selected_ids": [
        "ref_1",
        "ref_25",
        "ref_100"
    ]
}}
"""

DIAGRAM_PLANNER_PROMPT = """\
I am working on a task: given the 'Methodology' section of a paper, and the caption of the desired figure, automatically generate a corresponding illustrative diagram. I will input the text of the 'Methodology' section, the figure caption, and your output should be a detailed description of an illustrative figure that effectively represents the methods described in the text.

To help you understand the task better, and grasp the principles for generating such figures, I will also provide you with several examples. You should learn from these examples to provide your figure description.

** IMPORTANT: **
Your description should be as detailed as possible. Semantically, clearly describe each element and their connections. Formally, include various details such as background style (typically pure white or very light pastel), colors, line thickness, icon styles, etc. Remember: vague or unclear specifications will only make the generated figure worse, not better.

Your description should cover:
1. **Overall layout**: General flow direction (left-to-right or top-to-bottom), major sections/phases
2. **Components**: Each box, module, or element with its exact label
3. **Connections**: Arrows, data flows, and their directions
4. **Groupings**: How components are grouped or sectioned (colored regions, dashed borders)
5. **Labels and annotations**: Text labels, mathematical notations
6. **Input/Output**: What enters and exits the system
7. **Styling**: Background fills, color palettes (in natural language, e.g., "soft sky blue", "warm peach" — never hex codes), line weights, icon styles

## Methodology Section
{source_context}

## Figure Caption
{caption}

## Reference Examples
{examples}

Based on the methodology section, figure caption, and learning from the style and structure of the reference examples above, generate a comprehensive and detailed textual description of the methodology diagram.
"""

DIAGRAM_STYLIST_PROMPT = """\
You are a Lead Visual Designer for top-tier AI conferences (NeurIPS, ICML, ICLR, CVPR). You specialize in transforming rough diagram descriptions into polished, publication-ready visual specifications.

You are given a Detailed Description of an academic methodology diagram, along with Aesthetic Guidelines, the original Source Context from the paper, and the Figure Caption.

Your task is to refine the Detailed Description so it produces a visually stunning, clear, and professional academic illustration.

## 5 Crucial Instructions

1. **Preserve Aesthetics**: Maintain and enhance the visual quality. Use soft, muted pastel colors described in natural language (e.g., "soft sky blue", "warm peach", "light sage green"). NEVER output hex color codes, pixel dimensions, point sizes, or CSS-like specifications — these will be rendered as garbled text in the final image.

2. **Intervene Only When Necessary**: If the description already has strong visual design, make minimal changes. Do not rewrite for the sake of rewriting. Focus your edits on areas that genuinely need improvement.

3. **Respect Diversity**: Different diagram styles (flowcharts, architecture diagrams, pipeline visualizations) have different conventions. Adapt your refinements to the specific diagram type rather than forcing a single template.

4. **Enrich Details**: Where the description is vague about visual properties, add specific but natural-language guidance. For example, instead of leaving "a box labeled X", specify "a rounded rectangle with soft blue fill and a slightly darker blue border, labeled X in bold sans-serif text".

5. **Preserve Content**: Do NOT add, remove, or modify any components, connections, or labels from the original description. Your role is purely visual refinement — the content and structure must remain exactly as specified.

## Aesthetic Guidelines
{guidelines}

## Source Context
{source_context}

## Figure Caption
{caption}

## Current Description
{description}

Output ONLY the final polished Detailed Description. Do not include any conversational text, explanations, or preamble.
"""

DIAGRAM_VISUALIZER_PROMPT = """\
You are an expert scientific diagram illustrator. Generate high-quality scientific diagrams based on user requests. Note that do not include figure titles in the image.

CRITICAL: All text labels in the diagram must be rendered in clear, readable English. Use the EXACT label names specified in the description. Do not generate garbled, misspelled, or non-English text.

{description}
"""

DIAGRAM_CRITIC_PROMPT = """\
## ROLE

You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
Your task is to conduct a sanity check and provide a critique of the target diagram based on its content and presentation. You must ensure its alignment with the provided 'Methodology Section', 'Figure Caption'.

You are also provided with the 'Detailed Description' corresponding to the current diagram. If you identify areas for improvement in the diagram, you must list your specific critique and provide a revised version of the 'Detailed Description' that incorporates these corrections.

## CRITIQUE & REVISION RULES

1. Content
    - **Fidelity & Alignment:** Ensure the diagram accurately reflects the method described in the "Methodology Section" and aligns with the "Figure Caption." Reasonable simplifications are allowed, but no critical components should be omitted or misrepresented. Also, the diagram should not contain any hallucinated content. Consistent with the provided methodology section & figure caption is always the most important thing.
    - **Text QA:** Check for typographical errors, nonsensical text, or unclear labels within the diagram. Flag any garbled, misspelled, or non-English text. Flag any hex codes, pixel dimensions, or CSS values rendered as text. Suggest specific corrections.
    - **Validation of Examples:** Verify the accuracy of illustrative examples. If the diagram includes specific examples to aid understanding (e.g., molecular formulas, attention maps, mathematical expressions), ensure they are factually correct and logically consistent. If an example is incorrect, provide the correct version.
    - **Caption Exclusion:** Ensure the figure caption text (e.g., "Figure 1: Overview...") is **not** included within the image visual itself. The caption should remain separate.
2. Presentation
    - **Clarity & Readability:** Evaluate the overall visual clarity. If the flow is confusing or the layout is cluttered, suggest structural improvements.
    - **Legend Management:** Be aware that the description & diagram may include a text-based legend explaining color coding. Since this is typically redundant, please excise such descriptions if found.

** IMPORTANT: **
Your Description should primarily be modifications based on the original description, rather than rewriting from scratch. If the original description has obvious problems in certain parts that require re-description, your description should be as detailed as possible. Semantically, clearly describe each element and their connections. Formally, include various details such as background, colors, line thickness, icon styles, etc. Remember: vague or unclear specifications will only make the generated figure worse, not better.

## INPUT DATA

- **Methodology Section**: {source_context}
- **Figure Caption**: {caption}
- **Detailed Description**: {description}
- **Target Diagram**: [The generated figure is provided as an image]

## OUTPUT
Provide your response strictly in the following JSON format:
{{
    "critic_suggestions": ["specific actionable suggestion 1", "specific actionable suggestion 2"],
    "revised_description": "The complete revised description incorporating all suggested fixes. If no revision is needed, set to null."
}}

If the image is publication-ready with no issues, return:
{{
    "critic_suggestions": [],
    "revised_description": null
}}
"""

PLOT_RETRIEVER_PROMPT = """\
# Background & Goal
We are building an **AI system to automatically generate statistical plots**. Given a plot's raw data and the visual intent, the system needs to create a high-quality visualization that effectively presents the data.

To help the AI learn how to generate appropriate plots, we use a **few-shot learning approach**: we provide it with reference examples of similar plots. The AI will learn from these examples to understand what kind of plot to create for the target data.

# Your Task
**You are the Retrieval Agent.** Your job is to select the most relevant reference plots from a candidate pool that will serve as few-shot examples for the plot generation model.

You will receive:

- **Target Input:** The raw data and visual intent of the plot we need to generate
- **Candidate Pool:** Reference plots (each with raw data and visual intent)

You must select the **Top {num_examples} candidates** that would be most helpful as examples for teaching the AI how to create the target plot.

# Selection Logic (Data Type + Visual Intent)

Your goal is to find examples that match the Target in both **Data Characteristics** and **Plot Type**.

**1. Match Data Characteristics (Use Raw Data & Visual Intent):**
* What type of data is it? (e.g., categorical vs numerical, single series vs multi-series, temporal vs comparative).
* What are the data dimensions? (e.g., 1D, 2D, 3D).
* Select candidates with **similar data structures and characteristics**.
* *Why?* Different data types require different visualization approaches.

**2. Match Visual Intent (Use Visual Intent):**
* What type of plot is implied? (e.g., "bar chart", "scatter plot", "line chart", "pie chart", "heatmap", "radar chart").
* Select candidates with **similar plot types**.
* *Why?* A "bar chart" example is more useful for generating another bar chart than a "scatter plot" example, even if the data domains are similar.

**Ranking Priority:**

1. **Best Match:** Same Data Type AND Same Plot Type (e.g., Target is "multi-series line chart" -> Candidate is "multi-series line chart").
2. **Second Best:** Same Plot Type with compatible data (e.g., Target is "bar chart with 5 categories" -> Candidate is "bar chart with 6 categories").
3. **Avoid:** Different Plot Type (e.g., Target is "bar chart" -> Candidate is "pie chart"), unless there are no more candidates with the same plot type.

# Input Data

## Target Input

- **Visual Intent:** {caption}
- **Raw Data:** {source_context}

## Candidate Pool
{candidates}

# Output Format
Provide your output strictly in the following JSON format, containing only the **exact Plot IDs** of the Top {num_examples} selected plots:
{{
    "selected_ids": [
        "ref_0",
        "ref_25",
        "ref_100"
    ]
}}
"""

PLOT_PLANNER_PROMPT = """\
I am working on a task: given the raw data (typically in tabular or json format) and a visual intent of the desired plot, automatically generate a corresponding statistical plot that is both accurate and aesthetically pleasing. I will input the raw data and the plot visual intent, and your output should be a detailed description of an illustrative plot that effectively represents the data. Note that your description should include all the raw data points to be plotted.

To help you understand the task better, and grasp the principles for generating such plots, I will also provide you with several examples. You should learn from these examples to provide your plot description.

** IMPORTANT: **
Your description should be as detailed as possible. For content, explain the precise mapping of variables to visual channels (x, y, hue) and explicitly enumerate every raw data point's coordinate to be drawn to ensure accuracy. For presentation, specify the exact aesthetic parameters, including specific color codes, font sizes for all labels, line widths, marker dimensions, legend placement, and grid styles. You should learn from the examples' content presentation and aesthetic design (e.g., color schemes).

## Raw Data
{source_context}

## Visual Intent (Figure Caption)
{caption}

## Reference Examples
{examples}

Based on the raw data, visual intent, and learning from the style and structure of the reference examples above, generate a comprehensive and detailed textual description of the statistical plot.
"""

PLOT_STYLIST_PROMPT = """\
## ROLE

You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are provided with a preliminary description of a statistical plot to be generated. However, this description may lack specific aesthetic details, such as color palettes, background styling, and font choices.

Your task is to refine and enrich this description based on the provided [NeurIPS 2025 Style Guidelines] to ensure the final generated image is a high-quality, publication-ready plot that strictly adheres to the NeurIPS 2025 aesthetic standards.

**Crucial Instructions:**

1. **Enrich Details:** Focus on specifying visual attributes (colors, fonts, line styles, layout adjustments) defined in the guidelines.
2. **Preserve Content:** Do NOT alter the semantic content, logic, or quantitative results of the plot. Your job is purely aesthetic refinement, not content editing.
3. **Context Awareness:** Use the provided "Source Context" and "Figure Caption" to understand the emphasis of the plot, ensuring the style supports the content effectively.

## INPUT DATA

- **Detailed Description**: {description}
- **Style Guidelines**: {guidelines}
- **Source Context**: {source_context}
- **Figure Caption**: {caption}

## OUTPUT
Output ONLY the final polished Detailed Description. Do not include any conversational text or explanations.
"""

PLOT_VISUALIZER_PROMPT = """\
You are an expert statistical plot illustrator. Write code to generate high-quality statistical plots based on user requests.

Generate complete, executable Python code using matplotlib and/or seaborn to create the following statistical plot. The code should save the figure to the path specified by the OUTPUT_PATH variable.

## Plot Description
{description}

## Requirements
- Set OUTPUT_PATH variable at the top of the code
- Use plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
- Do NOT include plt.show() calls
- Publication-quality figure suitable for NeurIPS/ICML/ICLR
- Clean, minimal design (maximize data-ink ratio)
- Professional, colorblind-friendly color palette
- Clear axis labels with appropriate font sizes
- Legend that does not obstruct data
- High resolution (300 DPI minimum)
- Only output the Python code, nothing else
"""

PLOT_CRITIC_PROMPT = """\
## ROLE

You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
Your task is to conduct a sanity check and provide a critique of the target plot based on its content and presentation. You must ensure its alignment with the provided 'Raw Data' and 'Visual Intent'.

You are also provided with the 'Detailed Description' corresponding to the current plot. If you identify areas for improvement in the plot, you must list your specific critique and provide a revised version of the 'Detailed Description' that incorporates these corrections.

## CRITIQUE & REVISION RULES

1. Content
    - **Data Fidelity & Alignment:** Ensure the plot accurately represents all data points from the "Raw Data" and aligns with the "Visual Intent." All quantitative values must be correct. No data should be hallucinated, omitted, or misrepresented.
    - **Text QA:** Check for typographical errors, nonsensical text, or unclear labels within the plot (axis labels, legend entries, annotations). Suggest specific corrections.
    - **Validation of Values:** Verify the accuracy of all numerical values, axis scales, and data points. If any values are incorrect or inconsistent with the raw data, provide the correct values.
    - **Caption Exclusion:** Ensure the figure caption text (e.g., "Figure 1: Performance comparison...") is **not** included within the image visual itself. The caption should remain separate.
2. Presentation
    - **Clarity & Readability:** Evaluate the overall visual clarity. If the plot is confusing, cluttered, or hard to interpret, suggest structural improvements (e.g., better axis labeling, clearer legend, appropriate plot type).
    - **Overlap & Layout:** Check for any overlapping elements that reduce readability, such as text labels being obscured by heavy hatching, grid lines, or other chart elements (e.g., pie chart labels inside dark slices). If overlaps exist, suggest adjusting element positions (e.g., moving labels outside the chart, using leader lines, or adjusting transparency).
    - **Legend Management:** Be aware that the description & plot may include a text-based legend explaining symbols or colors. Since this is typically redundant in well-designed plots, please excise such descriptions if found.
3. Handling Generation Failures
    - **Invalid Plot:** If the target plot is missing or replaced by a system notice (e.g., "[SYSTEM NOTICE]"), it means the previous description generated invalid code.
    - **Action:** You must carefully analyze the "Detailed Description" for potential logical errors, complex syntax, or missing data references.
    - **Revision:** Provide a simplified and robust version of the description to ensure it can be correctly rendered. Do not just repeat the same description.

## INPUT DATA

- **Raw Data**: {source_context}
- **Visual Intent**: {caption}
- **Detailed Description**: {description}
- **Target Plot**: [The generated plot is provided as an image]

## OUTPUT
Provide your response strictly in the following JSON format:
{{
    "critic_suggestions": ["specific actionable suggestion 1", "specific actionable suggestion 2"],
    "revised_description": "The complete revised description incorporating all suggested fixes. If no revision is needed, set to null."
}}

If the plot is publication-ready with no issues, return:
{{
    "critic_suggestions": [],
    "revised_description": null
}}
"""

# ═══════════════════════════════════════════════════════════════════════════
# STYLE GUIDELINES
# ═══════════════════════════════════════════════════════════════════════════

METHODOLOGY_GUIDELINES = """\
# NeurIPS 2025 Method Diagram Aesthetics Guide

## 1. The "NeurIPS Look"

The prevailing aesthetic for 2025 is **"Soft Tech & Scientific Pastels."**
Gone are the days of harsh primary colors and sharp black boxes. The modern
NeurIPS diagram feels approachable yet precise. It utilizes high-value (
light) backgrounds to organize complexity, reserving saturation for the
most critical active elements. The vibe balances **clean modularity** (
clear separation of parts) with **narrative flow** (clear left-to-right
progression).

---

## 2. Detailed Style Options

### **A. Color Palettes**

*Design Philosophy: Use color to group logic, not just to decorate. Avoid
fully saturated backgrounds.*

**Background Fills (The "Zone" Strategy)**

*Used to encapsulate stages (e.g., "Pre-training phase") or environments.*

* **Most papers use:** Very light, desaturated pastels (Opacity ~10-15%).
* **Aesthetically pleasing options include:**
  * **Cream / Beige** (e.g., '#F5F5DC') - *Warm, academic feel.*
  * **Pale Blue / Ice** (e.g., '#E6F3FF') - *Clean, technical feel.*
  * **Mint / Sage** (e.g., '#E0F2F1') - *Soft, organic feel.*
  * **Pale Lavender** (e.g., '#F3E5F5') - *Distinctive, modern feel.*
* **Alternative (~20%):** White backgrounds with colored *dashed borders*
  for a high-contrast, minimalist look (common in theoretical papers).

**Functional Element Colors**

* **For "Active" Modules (Encoders, MLP, Attention):** Medium saturation
  is preferred.
  * *Common pairings:* Blue/Orange, Green/Purple, or Teal/Pink.
  * *Observation:* Colors are often used to distinguish **status**
    rather than component type:
    * **Trainable Elements:** Often Warm tones (Red, Orange, Deep Pink).
    * **Frozen/Static Elements:** Often Cool tones (Grey, Ice Blue, Cyan).
* **For Highlights/Results:** High saturation (Primary Red, Bright Gold)
  is strictly reserved for "Error/Loss," "Ground Truth," or the final
  output.

### **B. Shapes & Containers**

*Design Philosophy: "Softened Geometry." Sharp corners are for data; rounded
corners are for processes.*

**Core Components**

* **Process Nodes (The Standard):** Rounded Rectangles (Corner radius 5-10
  px). This is the dominant shape (~80%) for generic layers or steps.
* **Tensors & Data:**
  * **3D Stacks/Cuboids:** Used to imply depth/volume (e.g., B x H x W).
  * **Flat Squares/Grids:** Used for matrices, tokens, or attention maps.
* **Cylinders:** Exclusively reserved for Databases, Buffers, or Memory.

**Grouping & Hierarchy**

* **The "Macro-Micro" Pattern:** A solid, light-colored container
  represents the global view, with a specific module (e.g., "Attention
  Block") connected via lines to a "zoomed-in" detailed breakout box.
* **Borders:**
  * **Solid:** For physical components.
  * **Dashed:** Highly prevalent for indicating "Logical Stages,"
    "Optional Paths," or "Scopes."

### **C. Lines & Arrows**

*Design Philosophy: Line style dictates flow type.*

**Connector Styles**

* **Orthogonal / Elbow (Right Angles):** Most papers use this for **Network
  Architectures** (implies precision, matrices, and tensors).
* **Curved / Bezier:** Common choices include this for **System Logic,
  Feedback Loops, or High-Level Data Flow** (implies narrative and
  connection).

**Line Semantics**

* **Solid Black/Grey:** Standard data flow (Forward pass).
* **Dashed Lines:** Universally recognized as "Auxiliary Flow."
  * *Used for:* Gradient updates, Skip connections, or Loss calculations.
* **Integrated Math:** Standard operators (plus for Add, times for
  Concat/Multiply) are frequently placed *directly* on the line or
  intersection.

### **D. Typography & Icons**

*Design Philosophy: Strict separation between "Labeling" and "Math."*

**Typography**

* **Labels (Module Names):** **Sans-Serif** (Arial, Roboto, Helvetica).
  * *Style:* Bold for headers, Regular for details.
* **Variables (Math):** **Serif** (Times New Roman, LaTeX default).
  * *Rule:* If it is a variable in your equation (e.g., x, theta,
    L), it **must** be Serif and Italicized in the diagram.

**Iconography Options**

* **For Model State:**
  * *Trainable:* Fire, Lightning.
  * *Frozen:* Snowflake, Padlock, Stop Sign (Greyed out).
* **For Operations:**
  * *Inspection:* Magnifying Glass.
  * *Processing/Computation:* Gear, Monitor.
* **For Content:**
  * *Text/Prompt:* Document, Chat Bubble.
  * *Image:* Actual thumbnail of an image (not just a square).

### **E. Layout & Composition**

* **Flow direction:** Left-to-right for sequential pipelines; top-to-bottom
  for hierarchical architectures. Be consistent within a diagram.
* **Alignment:** All elements should snap to an implicit grid. No floating
  or randomly placed components.
* **Spacing:** Consistent gaps between elements. Components within the same
  group should be closer together than components in different groups.
* **Balance:** Distribute visual weight evenly. Avoid heavy clusters on one
  side with empty space on the other.
* **Whitespace:** Use whitespace intentionally to separate phases, stages,
  or conceptual groups. Whitespace is a design element, not wasted space.

---

## 3. Common Pitfalls (How to Look "Amateur")

* **The "PowerPoint Default" Look:** Using standard Blue/Orange presets
  with heavy black outlines.
* **Font Mixing:** Using Times New Roman for "Encoder" labels (makes the
  paper look dated to the 1990s).
* **Inconsistent Dimension:** Mixing flat 2D boxes and 3D isometric cubes
  without a clear reason (e.g., 2D for logic, 3D for tensors is fine;
  random mixing is not).
* **Primary Backgrounds:** Using saturated Yellow or Blue backgrounds for
  grouping (distracts from the content).
* **Ambiguous Arrows:** Using the same line style for "Data Flow" and
  "Gradient Flow."

---

## 4. Domain-Specific Styles

**If you are writing an AGENT / LLM Paper:**
* **Vibe:** Illustrative, Narrative, "Friendly.", Cartoony.
* **Key Elements:** Use "User Interface" aesthetics. Chat bubbles for
  prompts, document icons for retrieval.
* **Characters:** It is common to use cute 2D vector robots, human avatars,
  or emojis to humanize the agent's reasoning steps.

**If you are writing a COMPUTER VISION / 3D Paper:**
* **Vibe:** Spatial, Dense, Geometric.
* **Key Elements:** Frustums (camera cones), Ray lines, and Point Clouds.
* **Color:** Often uses RGB color coding to denote axes or channel
  correspondence. Use heatmaps (Rainbow/Viridis) to show activation.

**If you are writing a THEORETICAL / OPTIMIZATION Paper:**
* **Vibe:** Minimalist, Abstract, "Textbook."
* **Key Elements:** Focus on graph nodes (circles) and manifolds (planes/
  surfaces).
* **Color:** Restrained. Mostly Grayscale/Black/White with one highlight
  color (e.g., Gold or Blue). Avoid "cartoony" elements.

**If you are writing a GENERATIVE / LEARNING Paper:**
* **Vibe:** Dynamic, Process-oriented.
* **Key Elements:** Use noise/denoising visual metaphors, latent space
  representations, and distribution visualizations.
* **Color:** Gradual color transitions to indicate progressive refinement
  or generation stages.
"""

PLOT_GUIDELINES = """\
# NeurIPS 2025 Statistical Plot Aesthetics Guide

## 1. The "NeurIPS Look": A High-Level Overview

The prevailing aesthetic for 2025 is defined by **precision, accessibility,
and high contrast**. The "default" academic look has shifted away from
bare-bones styling toward a more graphic, publication-ready presentation.

* **Vibe:** Professional, clean, and information-dense.
* **Backgrounds:** There is a heavy bias toward **stark white backgrounds**
  for maximum contrast in print and PDF reading, though the "Seaborn-style"
  light grey background remains an accepted variant.
* **Accessibility:** A strong emphasis on distinguishing data not just by
  color, but by texture (patterns) and shape (markers) to support black-and-white
  printing and colorblind readers.

---

## 2. Detailed Style Options

### **Color Palettes**

* **Categorical Data:**
  * **Soft Pastels:** Matte, low-saturation colors (salmon, sky blue,
    mint, lavender) are frequently used to prevent visual fatigue.
  * **Muted Earth Tones:** "Academic" palettes using olive, beige, slate
    grey, and navy.
  * **High-Contrast Primaries:** Used sparingly when categories must be
    distinct (e.g., deep orange vs. vivid purple).
  * **Accessibility Mode:** A growing trend involves combining color
    with **geometric patterns** (hatches, dots, stripes) to differentiate
    categories.
* **Sequential & Heatmaps:**
  * **Perceptually Uniform:** "Viridis" (blue-to-yellow) and "Magma/
    Plasma" (purple-to-orange) are the standard.
  * **Diverging:** "Coolwarm" (blue-to-red) is used for positive/negative
    value splits.
  * **Avoid:** The traditional "Jet/Rainbow" scale is almost entirely absent.

### **Axes & Grids**

* **Grid Style:**
  * **Visibility:** Grid lines are almost rarely solid. Common choices
    include **fine dashed ('--')** or **dotted (':')** lines in light gray.
  * **Placement:** Grids are consistently rendered *behind* data
    elements (low Z-order).
* **Spines (Borders):**
  * **The "Boxed" Look:** A full enclosure (black spines on all 4 sides)
    is very common.
  * **The "Open" Look:** Removing the top and right spines for a
    minimalist appearance.
* **Ticks:**
  * **Style:** Ticks are generally subtle, facing inward, or removed
    entirely in favor of grid alignment.

### **Layout & Typography**

* **Typography:**
  * **Font Family:** Exclusively **Sans-Serif** (resembling Helvetica,
    Arial, or DejaVu Sans). Serif fonts are rarely used for labels.
  * **Label Rotation:** X-axis labels are rotated **45 degrees** only
    when necessary to prevent overlap; otherwise, horizontal orientation
    is preferred.
* **Legends:**
  * **Internal Placement:** Floating the legend *inside* the plot area (
    top-left or top-right) to maximize the "data-ink ratio."
  * **Top Horizontal:** Placing the legend in a single row above the
    plot title.
* **Annotations:**
  * **Direct Labeling:** Instead of forcing readers to reference a
    legend, text is often placed directly next to lines or on top of bars.

---

## 3. Type-Specific Guidelines

### **Bar Charts & Histograms**

* **Borders:** Two distinct styles are accepted:
  * **High-Definition:** Using **black outlines** around colored bars
    for a "comic-book" or high-contrast look.
  * **Borderless:** Solid color fills with no outline (often used with
    light grey backgrounds).
* **Grouping:** Bars are grouped tightly, with significant whitespace
  between categorical groups.
* **Error Bars:** Consistently styled with **black, flat caps**.

### **Line Charts**

* **Markers:** A critical observation: Lines almost always include **geometric
  markers** (circles, squares, diamonds) at data points, rather than just
  being smooth strokes.
* **Line Styles:** Use **dashed lines** ('--') for theoretical limits,
  baselines, or secondary data, and **solid lines** for primary experimental
  data.
* **Uncertainty:** Represented by semi-transparent **shaded bands** (
  confidence intervals) rather than simple vertical error bars.

### **Tree & Pie/Donut Charts**

* **Separators:** Thick **white borders** are standard to separate slices
  or treemap blocks.
* **Structure:** Thick **Donut charts** are preferred over traditional Pie
  charts.
* **Emphasis:** "Exploding" (detaching) a specific slice is a common
  technique to highlight a key statistic.

### **Scatter Plots**

* **Shape Coding:** Use different marker shapes (e.g., circles vs.
  triangles) to encode a categorical dimension alongside color.
* **Fills:** Markers are typically solid and fully opaque.
* **3D Plots:** Depth is emphasized by drawing "walls" with grids or using
  drop-lines to the "floor" of the plot.

### **Heatmaps**

* **Aspect Ratio:** Cells are almost strictly **square**.
* **Annotation:** Writing the exact value (in white or black text) **inside
  the cell** is highly preferred over relying solely on a color bar.
* **Borders:** Cells are often borderless (smooth gradient look) or
  separated by very thin white lines.

### **Radar Charts**

* **Fills:** The polygon area uses **translucent fills** (alpha ~0.2) to
  show grid lines underneath.
* **Perimeter:** The outer boundary is marked by a solid, darker line.

### **Miscellaneous**

* **Dot Plots:** Used as a modern alternative to bar charts; often styled
  as "lollipops" (dots connected to the axis by a thin line).

---

## 4. Common Pitfalls (What to Avoid)

* **The "Excel Default" Look:** Avoid heavy 3D effects on bars, shadow
  drops, or serif fonts (Times New Roman) on axes.
* **The "Rainbow" Map:** Avoid the Jet/Rainbow colormap; it is considered
  outdated and perceptually misleading.
* **Ambiguous Lines:** A line chart *without* markers can look ambiguous
  if data points are sparse; always add markers.
* **Over-reliance on Color:** Failing to use patterns or shapes to
  distinguish groups makes the plot inaccessible to colorblind readers.
* **Cluttered Grids:** Avoid solid black grid lines; they compete with the
  data. Always use light grey/dashed grids.
"""

# ═══════════════════════════════════════════════════════════════════════════
# API WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════


@retry(stop=stop_after_attempt(8), wait=wait_exponential(min=2, max=120))
def call_vlm(prompt, images=None, temperature=1.0, max_tokens=4096, json_mode=False):
    """Call Gemini VLM with text and optional images."""
    from google.genai import types

    client = _get_client()

    contents = []
    if images:
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            contents.append(
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")
            )
    contents.append(prompt)

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if json_mode:
        config.response_mime_type = "application/json"

    response = client.models.generate_content(
        model=VLM_MODEL,
        contents=contents,
        config=config,
    )
    return response.text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def generate_image(prompt, width=1792, height=1024):
    """Generate an image using Gemini image generation."""
    from google.genai import types

    client = _get_client()

    # Compute aspect ratio
    ratio = width / height
    if ratio > 1.5:
        aspect = "16:9"
    elif ratio > 1.2:
        aspect = "3:2"
    elif ratio < 0.67:
        aspect = "9:16"
    elif ratio < 0.83:
        aspect = "2:3"
    else:
        aspect = "1:1"

    # Compute image size
    max_dim = max(width, height)
    if max_dim <= 1024:
        size = "1K"
    elif max_dim <= 2048:
        size = "2K"
    else:
        size = "4K"

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=aspect,
            image_size=size,
        ),
    )

    response = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=prompt,
        config=config,
    )

    # Extract image from response
    parts = None
    if getattr(response, "candidates", None):
        parts = response.candidates[0].content.parts
    else:
        parts = getattr(response, "parts", None)

    if not parts:
        raise ValueError("Gemini image response had no content parts.")

    for part in parts:
        if hasattr(part, "as_image"):
            try:
                return part.as_image()
            except Exception:
                pass
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            data = inline.data
            image_bytes = base64.b64decode(data) if isinstance(data, str) else data
            return Image.open(BytesIO(image_bytes))

    raise ValueError("Gemini image response did not contain image data.")


# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE LOADING
# ═══════════════════════════════════════════════════════════════════════════


def load_references(reference_dir):
    """Load reference examples from an index.json file.

    Returns a list of dicts with keys: id, source_context, caption, image_path, category.
    """
    ref_path = Path(reference_dir)
    index_file = ref_path / "index.json"
    if not index_file.exists():
        print(f"Warning: No reference index found at {index_file}")
        return []

    with open(index_file, encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for item in data.get("examples", []):
        image_path = item.get("image_path", "")
        if image_path and not Path(image_path).is_absolute():
            image_path = str(ref_path / image_path)
        examples.append({
            "id": item["id"],
            "source_context": item["source_context"],
            "caption": item["caption"],
            "image_path": image_path,
            "category": item.get("category"),
        })

    print(f"Loaded {len(examples)} reference examples from {reference_dir}")
    return examples


# ═══════════════════════════════════════════════════════════════════════════
# AGENT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════


def retrieve(source_context, caption, candidates, num_examples=10, mode="diagram"):
    """Select the most relevant reference examples via VLM."""
    if not candidates:
        print("Warning: No reference candidates available")
        return []

    if len(candidates) <= num_examples:
        return candidates

    # Format candidates
    lines = []
    for i, c in enumerate(candidates):
        lines.append(
            f"Candidate Paper {i + 1}:\n"
            f"- **Paper ID:** {c['id']}\n"
            f"- **Caption:** {c['caption']}\n"
            f"- **Methodology section:** {c['source_context'][:300]}...\n"
        )
    candidates_text = "\n".join(lines)

    template = DIAGRAM_RETRIEVER_PROMPT if mode == "diagram" else PLOT_RETRIEVER_PROMPT
    prompt = template.format(
        source_context=source_context,
        caption=caption,
        candidates=candidates_text,
        num_examples=num_examples,
    )

    print(f"[Retriever] Selecting top {num_examples} from {len(candidates)} candidates...")
    response = call_vlm(prompt, temperature=0.3, json_mode=True)

    # Parse response
    try:
        data = json.loads(response)
        selected_ids = (
            data.get("selected_ids")
            or data.get("top_10_papers")
            or data.get("top_10_plots")
            or []
        )
    except json.JSONDecodeError:
        print("Warning: Failed to parse retriever response, using all candidates")
        return candidates[:num_examples]

    id_to_example = {c["id"]: c for c in candidates}
    selected = []
    for eid in selected_ids:
        if eid in id_to_example:
            selected.append(id_to_example[eid])

    print(f"[Retriever] Selected {len(selected)} examples")
    return selected[:num_examples]


def plan(source_context, caption, examples, mode="diagram"):
    """Generate a detailed textual description using in-context learning."""
    # Format examples text
    if not examples:
        examples_text = "(No reference examples available. Generate based on source context alone.)"
    else:
        lines = []
        img_index = 0
        for i, ex in enumerate(examples, 1):
            has_image = ex.get("image_path") and Path(ex["image_path"]).exists()
            image_ref = ""
            if has_image:
                img_index += 1
                image_ref = f"\n**Diagram**: [See reference image {img_index} above]"
            lines.append(
                f"### Example {i}\n"
                f"**Caption**: {ex['caption']}\n"
                f"**Source Context**: {ex['source_context'][:500]}"
                f"{image_ref}\n"
            )
        examples_text = "\n".join(lines)

    # Load reference images
    example_images = []
    for ex in examples:
        if ex.get("image_path") and Path(ex["image_path"]).exists():
            try:
                img = Image.open(ex["image_path"]).convert("RGB")
                example_images.append(img)
            except Exception as e:
                print(f"Warning: Failed to load reference image {ex['image_path']}: {e}")

    template = DIAGRAM_PLANNER_PROMPT if mode == "diagram" else PLOT_PLANNER_PROMPT
    prompt = template.format(
        source_context=source_context,
        caption=caption,
        examples=examples_text,
    )

    print(f"[Planner] Generating description ({len(example_images)} reference images)...")
    description = call_vlm(
        prompt,
        images=example_images if example_images else None,
        temperature=0.7,
        max_tokens=4096,
    )

    print(f"[Planner] Generated description ({len(description)} chars)")
    return description


def style(description, source_context, caption, mode="diagram"):
    """Refine description with aesthetic guidelines."""
    guidelines = METHODOLOGY_GUIDELINES if mode == "diagram" else PLOT_GUIDELINES

    template = DIAGRAM_STYLIST_PROMPT if mode == "diagram" else PLOT_STYLIST_PROMPT
    prompt = template.format(
        description=description,
        guidelines=guidelines,
        source_context=source_context,
        caption=caption,
    )

    print("[Stylist] Refining description...")
    optimized = call_vlm(prompt, temperature=0.5, max_tokens=4096)

    print(f"[Stylist] Refined description ({len(optimized)} chars)")
    return optimized


def visualize(description, mode="diagram", raw_data=None, output_path=None, iteration=1):
    """Generate an image from a description.

    For diagrams: uses Gemini image generation.
    For plots: generates and executes matplotlib code.
    """
    if mode == "plot":
        return _generate_plot(description, raw_data, output_path, iteration)
    else:
        return _generate_diagram(description, output_path, iteration)


def _generate_diagram(description, output_path, iteration):
    """Generate a methodology diagram using image generation."""
    prompt = DIAGRAM_VISUALIZER_PROMPT.format(description=description)

    print(f"[Visualizer] Generating diagram (iteration {iteration})...")
    image = generate_image(prompt, width=1792, height=1024)

    if output_path is None:
        output_path = f"diagram_iter_{iteration}.png"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"[Visualizer] Saved to {output_path}")
    return output_path


def _generate_plot(description, raw_data, output_path, iteration):
    """Generate a statistical plot by generating and executing matplotlib code."""
    full_description = description
    if raw_data:
        full_description += f"\n\n## Raw Data\n```json\n{json.dumps(raw_data, indent=2)}\n```"

    code_prompt = PLOT_VISUALIZER_PROMPT.format(description=full_description)

    print(f"[Visualizer] Generating plot code (iteration {iteration})...")
    code_response = call_vlm(code_prompt, temperature=0.3, max_tokens=4096)

    # Extract code from response
    code = _extract_code(code_response)

    if output_path is None:
        output_path = f"plot_iter_{iteration}.png"

    # Execute the code
    success = _execute_plot_code(code, output_path)
    if not success:
        print("[Visualizer] Plot code execution failed, creating placeholder")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        placeholder = Image.new("RGB", (1024, 768), color=(255, 255, 255))
        placeholder.save(output_path)

    return output_path


def _extract_code(response):
    """Extract Python code from a VLM response."""
    if "```python" in response:
        start = response.index("```python") + len("```python")
        end = response.index("```", start)
        return response[start:end].strip()
    elif "```" in response:
        start = response.index("```") + 3
        end = response.index("```", start)
        return response[start:end].strip()
    return response.strip()


def _execute_plot_code(code, output_path):
    """Execute matplotlib code in a subprocess."""
    # Strip any OUTPUT_PATH assignments from generated code
    code = re.sub(r'^OUTPUT_PATH\s*=\s*["\'].*["\']\s*$', "", code, flags=re.MULTILINE)

    # Inject the output path
    full_code = f'OUTPUT_PATH = "{output_path}"\n{code}'

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"[Visualizer] Plot code error: {result.stderr[:500]}")
            return False
        return Path(output_path).exists()
    except subprocess.TimeoutExpired:
        print("[Visualizer] Plot code timed out")
        return False
    finally:
        Path(temp_path).unlink(missing_ok=True)


def critique(image_path, description, source_context, caption, mode="diagram"):
    """Evaluate a generated image and provide revision feedback.

    Returns a dict with keys: critic_suggestions (list), revised_description (str or None).
    """
    image = Image.open(image_path).convert("RGB")

    template = DIAGRAM_CRITIC_PROMPT if mode == "diagram" else PLOT_CRITIC_PROMPT
    prompt = template.format(
        source_context=source_context,
        caption=caption,
        description=description,
    )

    print("[Critic] Evaluating image...")
    response = call_vlm(prompt, images=[image], temperature=0.3, max_tokens=4096, json_mode=True)

    try:
        data = json.loads(response)
        result = {
            "critic_suggestions": data.get("critic_suggestions", []),
            "revised_description": data.get("revised_description"),
        }
    except (json.JSONDecodeError, KeyError):
        print("Warning: Failed to parse critic response")
        result = {"critic_suggestions": [], "revised_description": None}

    needs_revision = len(result["critic_suggestions"]) > 0
    if needs_revision:
        summary = "; ".join(result["critic_suggestions"][:3])
        print(f"[Critic] Suggestions: {summary}")
    else:
        print("[Critic] No issues found — publication-ready!")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


def generate(source_context, caption, reference_dir=None, mode="diagram",
             iterations=3, output_dir="outputs", raw_data=None):
    """Run the full generation pipeline.

    Args:
        source_context: Methodology text or raw data.
        caption: Figure caption / communicative intent.
        reference_dir: Path to reference set directory with index.json.
        mode: "diagram" or "plot".
        iterations: Number of refinement iterations (default 3).
        output_dir: Base output directory.
        raw_data: Raw data dict for plot mode.

    Returns:
        Path to the final output image.
    """
    total_start = time.perf_counter()

    # Create run directory
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    run_id = f"run_{ts}_{short_uuid}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PaperBanana Lite — {mode} generation")
    print(f"Run ID: {run_id}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}\n")

    # Load references
    candidates = []
    if reference_dir:
        candidates = load_references(reference_dir)
    else:
        # Try default location
        default_ref = Path("data/reference_sets")
        if default_ref.exists():
            candidates = load_references(str(default_ref))

    # ── Phase 1: Linear Planning ─────────────────────────────────

    print("\n--- Phase 1: Linear Planning ---\n")

    # Step 1: Retriever
    t0 = time.perf_counter()
    examples = retrieve(source_context, caption, candidates,
                        num_examples=NUM_RETRIEVAL_EXAMPLES, mode=mode)
    print(f"    ({time.perf_counter() - t0:.1f}s)\n")

    # Step 2: Planner
    t0 = time.perf_counter()
    description = plan(source_context, caption, examples, mode=mode)
    print(f"    ({time.perf_counter() - t0:.1f}s)\n")

    # Step 3: Stylist
    t0 = time.perf_counter()
    description = style(description, source_context, caption, mode=mode)
    print(f"    ({time.perf_counter() - t0:.1f}s)\n")

    # Save planning outputs
    planning_data = {
        "retrieved_examples": [e["id"] for e in examples],
        "initial_description": description,
    }
    with open(run_dir / "planning.json", "w") as f:
        json.dump(planning_data, f, indent=2)

    # ── Phase 2: Iterative Refinement ─────────────────────────────

    print(f"\n--- Phase 2: Iterative Refinement (up to {iterations} iterations) ---\n")

    current_description = description
    final_image_path = None

    for i in range(iterations):
        print(f"\n  Iteration {i + 1}/{iterations}")
        print(f"  {'-'*40}\n")

        # Step 4: Visualizer
        t0 = time.perf_counter()
        image_path = visualize(
            current_description,
            mode=mode,
            raw_data=raw_data,
            output_path=str(run_dir / f"iter_{i + 1}.png"),
            iteration=i + 1,
        )
        print(f"    ({time.perf_counter() - t0:.1f}s)\n")

        final_image_path = image_path

        # Step 5: Critic
        t0 = time.perf_counter()
        crit = critique(image_path, current_description, source_context, caption, mode=mode)
        print(f"    ({time.perf_counter() - t0:.1f}s)\n")

        # Save iteration details
        iter_data = {
            "iteration": i + 1,
            "description": current_description,
            "critic_suggestions": crit["critic_suggestions"],
            "revised_description": crit["revised_description"],
        }
        with open(run_dir / f"iter_{i + 1}_details.json", "w") as f:
            json.dump(iter_data, f, indent=2)

        # Check if revision needed
        needs_revision = len(crit["critic_suggestions"]) > 0
        if needs_revision and crit["revised_description"]:
            print("  -> Revision needed, updating description for next iteration")
            current_description = crit["revised_description"]
        else:
            print("  -> No further revision needed, stopping early")
            break

    # Copy final image
    final_output = str(run_dir / "final_output.png")
    shutil.copy2(final_image_path, final_output)

    total_seconds = time.perf_counter() - total_start

    print(f"\n{'='*60}")
    print("Generation complete!")
    print(f"Final output: {final_output}")
    print(f"Total time: {total_seconds:.1f}s")
    print(f"{'='*60}\n")

    return final_output


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        prog="paperbanana_lite",
        description="PaperBanana Lite — single-file academic illustration generation",
    )
    subparsers = parser.add_subparsers(dest="command", help="Generation mode")

    # --- generate subcommand (diagrams) ---
    gen_parser = subparsers.add_parser("generate", help="Generate a methodology diagram")
    gen_parser.add_argument("--input", required=True,
                            help="Path to text file with methodology section")
    gen_parser.add_argument("--caption", required=True,
                            help="Figure caption / communicative intent")
    gen_parser.add_argument("--reference-dir", default=None,
                            help="Path to reference set directory (default: data/reference_sets)")
    gen_parser.add_argument("--iterations", type=int, default=3,
                            help="Number of refinement iterations (default: 3)")
    gen_parser.add_argument("--output-dir", default="outputs",
                            help="Base output directory (default: outputs)")

    # --- plot subcommand ---
    plot_parser = subparsers.add_parser("plot", help="Generate a statistical plot")
    plot_parser.add_argument("--data", required=True,
                             help="Path to JSON file with raw data")
    plot_parser.add_argument("--intent", required=True,
                             help="Visual intent / figure caption")
    plot_parser.add_argument("--reference-dir", default=None,
                             help="Path to reference set directory (default: data/reference_sets)")
    plot_parser.add_argument("--iterations", type=int, default=3,
                             help="Number of refinement iterations (default: 3)")
    plot_parser.add_argument("--output-dir", default="outputs",
                             help="Base output directory (default: outputs)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        # Read methodology text
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        source_context = input_path.read_text(encoding="utf-8")

        generate(
            source_context=source_context,
            caption=args.caption,
            reference_dir=args.reference_dir,
            mode="diagram",
            iterations=args.iterations,
            output_dir=args.output_dir,
        )

    elif args.command == "plot":
        # Read raw data
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"Error: Data file not found: {data_path}", file=sys.stderr)
            sys.exit(1)

        with open(data_path, encoding="utf-8") as f:
            raw_data = json.load(f)

        # Use the JSON content as source_context too
        source_context = json.dumps(raw_data, indent=2)

        generate(
            source_context=source_context,
            caption=args.intent,
            reference_dir=args.reference_dir,
            mode="plot",
            iterations=args.iterations,
            output_dir=args.output_dir,
            raw_data=raw_data,
        )


if __name__ == "__main__":
    main()
