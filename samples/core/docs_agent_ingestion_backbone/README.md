# Docs Agent Ingestion Backbone (KFP)

This sample scaffolds the proposed deterministic ingestion backbone as a KFP v2
pipeline:

1. `fetch_sources`
2. `filter_manifest`
3. `extract_text_docs` (parallel track A)
4. `process_visual_artifacts` (parallel track B)
5. `structure_platform_knowledge`
6. `validate_ingestion_output`
7. `persist_and_optimize`

Current status:

- Stage 1 implemented:
  - `fetch_sources`: shallow clone + commit provenance metadata.
  - `filter_manifest`: repository scan + text/visual manifests.
- Stage 2 implemented:
  - `extract_text_docs`: header-aware Markdown chunking, code block extraction,
    and image reference extraction.
  - `process_visual_artifacts`: fidelity-aware parsing:
    `.mmd` (code-based), `.drawio.xml` (structured XML), `.svg` (text/title),
    `.png` (OCR fallback status marker).
- Stage 3-5 are wired and functional but still baseline:
  - Stage 3 implemented:
    - builds an in-memory visual lookup map keyed by normalized visual path.
    - resolves chunk-level relative image references to repo-root-relative
      paths.
    - injects matched visual artifacts into each chunk's `visual_context`.
    - enriches artifacts with deterministic provenance and component metadata.
  - Stage 4 implemented with three-layer guardrails:
    - structural integrity: required field/schema checks (drop invalid records).
    - navigational integrity: markdown internal link resolution and existence
      checks against `source_volume`; broken links are retained and tagged.
    - semantic quality: stub/noise filtering (`TODO`, `WIP`, `Coming Soon`,
      short content) and stability tagging (`Deprecated`, `Alpha`,
      `Experimental`).
    - emits `validation_report` with summary metrics and sample findings.
  - Stage 5 implemented:
    - builds flattened retrieval text from `header_path`, `text_content`, and
      `visual_context`.
    - injects deterministic metadata payload (component/source/provenance/quality).
    - emits deterministic `record_id` and `source_key`.
    - emits idempotent persistence contract with `operation: upsert` and
      `delete_scope` keyed by `source_key`.

Compile locally:

```bash
source .venv/bin/activate
python samples/core/docs_agent_ingestion_backbone/docs_agent_ingestion_backbone.py
```

Notes:

- Running the compile command generates
  `samples/core/docs_agent_ingestion_backbone/docs_agent_ingestion_backbone.py.yaml`.

Run locally and inspect final output:

```python
from kfp import local
from samples.core.docs_agent_ingestion_backbone.docs_agent_ingestion_backbone import (
    docs_agent_ingestion_backbone_pipeline,
)

# use_venv=False avoids per-task pip installs in restricted/offline environments
local.init(runner=local.SubprocessRunner(use_venv=False))
docs_agent_ingestion_backbone_pipeline(
    repo_url='file:///home/shreeharsh157/Desktop/pipelines',
    git_ref='master',
)
```

After a successful run, artifacts are written under:

- `local_outputs/docs-agent-ingestion-backbone-*/persist-and-optimize/retrieval_ready_records`
- `local_outputs/docs-agent-ingestion-backbone-*/validate-ingestion-output/validation_report`

Quick check:

```bash
LATEST_RUN="$(ls -dt local_outputs/docs-agent-ingestion-backbone-* | head -n 1)"
wc -l "$LATEST_RUN/persist-and-optimize/retrieval_ready_records"
sed -n '1,2p' "$LATEST_RUN/persist-and-optimize/retrieval_ready_records"
```
