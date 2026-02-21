#!/usr/bin/env python3
# Copyright 2026 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deterministic ingestion backbone sample for a docs-agent style RAG system.

This sample implements a 5-stage KFP DAG:
1) Acquisition: fetch_sources, filter_manifest
2) Transformation: extract_text_docs, process_visual_artifacts
3) Integration: structure_platform_knowledge
4) Validation: validate_ingestion_output
5) Retrieval preparation: persist_and_optimize

The goal is to keep each stage independently testable and to emit explicit,
metadata-rich artifacts between stages.
"""

import os

from kfp import compiler
from kfp import dsl
from kfp.dsl import Artifact
from kfp.dsl import Dataset
from kfp.dsl import Input
from kfp.dsl import Output

_KFP_PACKAGE_PATH = os.getenv('KFP_PACKAGE_PATH')


@dsl.component(kfp_package_path=_KFP_PACKAGE_PATH)
def fetch_sources(
    repo_url: str,
    git_ref: str,
    source_volume: Output[Artifact],
    provenance_metadata: Output[Dataset],
):
    """Stage 1A: Clone repository snapshot and emit provenance metadata.

    Inputs:
    - repo_url: source repository URL
    - git_ref: branch/tag to snapshot

    Outputs:
    - source_volume: directory artifact containing cloned repository at /repo
    - provenance_metadata: JSON dataset with commit hash and ingestion metadata
    """
    import datetime
    import json
    import os
    import shutil
    import subprocess

    source_root = source_volume.path
    if os.path.exists(source_root):
        shutil.rmtree(source_root)
    os.makedirs(source_root, exist_ok=True)

    clone_target = os.path.join(source_root, 'repo')
    subprocess.run(
        ['git', 'clone', '--depth', '1', '--branch', git_ref, repo_url, clone_target],
        check=True,
    )

    commit_hash = subprocess.check_output(
        ['git', '-C', clone_target, 'rev-parse', 'HEAD'],
        text=True,
    ).strip()

    metadata = {
        'repo_url': repo_url,
        'git_ref': git_ref,
        'commit_hash': commit_hash,
        'snapshot_root': 'repo',
        'ingested_at_utc': datetime.datetime.utcnow().replace(
            microsecond=0).isoformat() + 'Z',
    }
    with open(provenance_metadata.path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


@dsl.component(kfp_package_path=_KFP_PACKAGE_PATH)
def filter_manifest(
    source_volume: Input[Artifact],
    text_file_manifest: Output[Dataset],
    visual_file_manifest: Output[Dataset],
):
    """Stage 1B: Scan repository snapshot and classify text vs visual assets.

    Inputs:
    - source_volume (from fetch_sources)

    Outputs:
    - text_file_manifest: JSON list of text files (.md/.rst/.txt)
    - visual_file_manifest: JSON list of diagram/image files
    """
    import json
    import os
    from pathlib import Path

    repo_root = os.path.join(source_volume.path, 'repo')
    if not os.path.isdir(repo_root):
        raise FileNotFoundError(f'Repository snapshot not found: {repo_root}')

    excluded_dirs = {
        '.git',
        '.idea',
        '.venv',
        'venv',
        'node_modules',
        'vendor',
        'build',
        'dist',
    }
    text_exts = {'.md', '.rst', '.txt'}
    visual_exts = {'.png', '.svg', '.drawio.xml', '.mmd'}

    text_entries = []
    visual_entries = []

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for filename in files:
            full_path = os.path.join(root, filename)
            rel_path = Path(os.path.relpath(full_path, repo_root)).as_posix()

            ext = ''.join(Path(filename).suffixes).lower()
            if ext in text_exts:
                text_entries.append({'path': rel_path, 'kind': 'text', 'ext': ext})
            if ext in visual_exts:
                visual_entries.append({'path': rel_path, 'kind': 'visual', 'ext': ext})

    text_entries.sort(key=lambda x: x['path'])
    visual_entries.sort(key=lambda x: x['path'])

    with open(text_file_manifest.path, 'w', encoding='utf-8') as f:
        json.dump(text_entries, f, indent=2)
    with open(visual_file_manifest.path, 'w', encoding='utf-8') as f:
        json.dump(visual_entries, f, indent=2)


@dsl.component(kfp_package_path=_KFP_PACKAGE_PATH)
def extract_text_docs(
    source_volume: Input[Artifact],
    text_file_manifest: Input[Dataset],
    structured_text_chunks: Output[Dataset],
):
    """Stage 2A: Convert text assets into structured, context-aware chunks.

    Behavior:
    - Markdown: header-aware chunking with fenced code block extraction
    - Non-Markdown text: single chunk fallback
    - Extracts image references for Stage 3 visual context injection
    """
    import json
    import os
    import re

    repo_root = os.path.join(source_volume.path, 'repo')
    with open(text_file_manifest.path, 'r', encoding='utf-8') as f:
        manifests = json.load(f)

    heading_pattern = re.compile(r'^(#{1,6})\s+(.*\S)\s*$')
    image_pattern = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
    fence_pattern = re.compile(r'^(```|~~~)\s*([a-zA-Z0-9_+-]*)\s*$')

    def emit_chunk(out_handle, source_path, chunk_index, header_path, text_lines,
                   code_blocks, image_refs):
        text_content = '\n'.join(text_lines).strip()
        if not text_content and not code_blocks and not image_refs:
            return chunk_index
        record = {
            'chunk_id': f'{source_path}::{chunk_index}',
            'source_path': source_path,
            'header_path': list(header_path),
            'text_content': text_content,
            'code_blocks': code_blocks,
            'image_references': sorted(image_refs),
        }
        out_handle.write(json.dumps(record) + '\n')
        return chunk_index + 1

    with open(structured_text_chunks.path, 'w', encoding='utf-8') as out:
        for item in manifests:
            rel_path = item['path']
            file_path = os.path.join(repo_root, rel_path)
            if not os.path.isfile(file_path):
                continue
            with open(file_path, 'r', encoding='utf-8', errors='replace') as src:
                lines = src.read().splitlines()

            lowered = rel_path.lower()
            if lowered.endswith('.md'):
                header_path = []
                chunk_index = 0
                text_lines = []
                image_refs = set()
                code_blocks = []
                in_fence = False
                fence_marker = None
                fence_lang = ''
                fence_lines = []

                for line in lines:
                    if in_fence:
                        close_match = fence_pattern.match(line)
                        if close_match and close_match.group(1) == fence_marker:
                            code_blocks.append({
                                'language': fence_lang,
                                'content': '\n'.join(fence_lines).strip(),
                            })
                            in_fence = False
                            fence_marker = None
                            fence_lang = ''
                            fence_lines = []
                            continue
                        fence_lines.append(line)
                        continue

                    fence_match = fence_pattern.match(line)
                    if fence_match:
                        in_fence = True
                        fence_marker = fence_match.group(1)
                        fence_lang = fence_match.group(2) or ''
                        fence_lines = []
                        continue

                    heading_match = heading_pattern.match(line)
                    if heading_match:
                        chunk_index = emit_chunk(
                            out_handle=out,
                            source_path=rel_path,
                            chunk_index=chunk_index,
                            header_path=header_path,
                            text_lines=text_lines,
                            code_blocks=code_blocks,
                            image_refs=image_refs,
                        )
                        level = len(heading_match.group(1))
                        title = heading_match.group(2).strip()
                        header_path = header_path[:level - 1] + [title]
                        text_lines = []
                        image_refs = set()
                        code_blocks = []
                        continue

                    text_lines.append(line)
                    image_refs.update(image_pattern.findall(line))

                if in_fence:
                    code_blocks.append({
                        'language': fence_lang,
                        'content': '\n'.join(fence_lines).strip(),
                    })
                emit_chunk(
                    out_handle=out,
                    source_path=rel_path,
                    chunk_index=chunk_index,
                    header_path=header_path,
                    text_lines=text_lines,
                    code_blocks=code_blocks,
                    image_refs=image_refs,
                )
            else:
                text = '\n'.join(lines).strip()
                record = {
                    'chunk_id': f'{rel_path}::0',
                    'source_path': rel_path,
                    'header_path': [],
                    'text_content': text,
                    'code_blocks': [],
                    'image_references': sorted(image_pattern.findall(text)),
                }
                out.write(json.dumps(record) + '\n')


@dsl.component(kfp_package_path=_KFP_PACKAGE_PATH)
def process_visual_artifacts(
    source_volume: Input[Artifact],
    visual_file_manifest: Input[Dataset],
    structured_visual_artifacts: Output[Dataset],
):
    """Stage 2B: Parse visual assets using a fidelity-aware strategy.

    Fidelity order:
    - .mmd (code-based diagrams)
    - .drawio.xml (structured source)
    - .svg / .png fallback (best-effort extraction)

    Output is normalized JSONL containing labels/relationships/parse status.
    """
    import json
    import os
    import re

    try:
        from defusedxml import ElementTree as ET
    except ImportError:
        from xml.etree import ElementTree as ET

    repo_root = os.path.join(source_volume.path, 'repo')
    with open(visual_file_manifest.path, 'r', encoding='utf-8') as f:
        manifests = json.load(f)

    def parse_mermaid(raw_text):
        relationships = []
        labels = set()
        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if '-->' in stripped or '---' in stripped:
                relationships.append(stripped)
            labels.update(re.findall(r'\[([^\]]+)\]|\(([^)]+)\)|\{([^}]+)\}', stripped))
        flat_labels = []
        for triple in labels:
            value = next((entry for entry in triple if entry), '').strip()
            if value:
                flat_labels.append(value)
        return sorted(set(flat_labels)), relationships

    def parse_drawio_xml(file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        labels = set()
        relationships = []
        for elem in root.iter():
            raw_label = elem.attrib.get('value') or elem.attrib.get('label')
            if raw_label:
                clean = re.sub(r'<[^>]+>', ' ', raw_label)
                clean = re.sub(r'\s+', ' ', clean).strip()
                if clean:
                    labels.add(clean)
            if elem.attrib.get('edge') == '1':
                src = elem.attrib.get('source', '')
                dst = elem.attrib.get('target', '')
                relationships.append({'source': src, 'target': dst})
        return sorted(labels), relationships

    def parse_svg(file_path):
        labels = set()
        relationships = []
        tree = ET.parse(file_path)
        root = tree.getroot()
        for elem in root.iter():
            tag = elem.tag.lower()
            if tag.endswith('text') and elem.text:
                text = elem.text.strip()
                if text:
                    labels.add(text)
            if tag.endswith('title') and elem.text:
                title = elem.text.strip()
                if title:
                    labels.add(title)
            if tag.endswith('line') or tag.endswith('path'):
                relationships.append(tag.rsplit('}', 1)[-1])
        return sorted(labels), relationships

    with open(structured_visual_artifacts.path, 'w', encoding='utf-8') as out:
        for item in manifests:
            source_path = item['path']
            full_path = os.path.join(repo_root, source_path)
            lowered = source_path.lower()

            labels = []
            relationships = []
            flow_descriptions = []
            fidelity = 'pixel-fallback'
            parse_status = 'ok'

            if not os.path.isfile(full_path):
                parse_status = 'source_file_missing'
            elif lowered.endswith('.mmd'):
                fidelity = 'code-based'
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                labels, relationships = parse_mermaid(content)
                flow_descriptions = relationships[:]
            elif lowered.endswith('.drawio.xml'):
                fidelity = 'structured-diagram'
                try:
                    labels, relationships = parse_drawio_xml(full_path)
                    flow_descriptions = [
                        f"edge {edge.get('source')} -> {edge.get('target')}"
                        for edge in relationships
                    ]
                except Exception:
                    parse_status = 'drawio_parse_failed'
            elif lowered.endswith('.svg'):
                fidelity = 'pixel-fallback'
                try:
                    labels, relationships = parse_svg(full_path)
                    flow_descriptions = ['svg text extraction only']
                except Exception:
                    parse_status = 'svg_parse_failed'
            elif lowered.endswith('.png'):
                fidelity = 'pixel-fallback'
                parse_status = 'ocr_not_configured'
                flow_descriptions = [
                    'OCR fallback unavailable in this component runtime.',
                ]

            record = {
                'source_path': source_path,
                'fidelity': fidelity,
                'parse_status': parse_status,
                'labels': labels,
                'relationships': relationships,
                'flow_descriptions': flow_descriptions,
            }
            out.write(json.dumps(record) + '\n')


@dsl.component(kfp_package_path=_KFP_PACKAGE_PATH)
def structure_platform_knowledge(
    structured_text_chunks: Input[Dataset],
    structured_visual_artifacts: Input[Dataset],
    provenance_metadata: Input[Dataset],
    structured_knowledge_artifacts: Output[Dataset],
):
    """Stage 3: Join structured text with diagram-derived knowledge.

    Core logic:
    - Builds in-memory index of visual artifacts by normalized source path
    - Resolves text chunk image references relative to source document
    - Injects matched visuals into chunk-level visual_context
    - Adds deterministic provenance/component metadata
    """
    import json
    import posixpath
    import re

    with open(provenance_metadata.path, 'r', encoding='utf-8') as f:
        provenance = json.load(f)

    def normalize_repo_path(base_doc_path, ref_path):
        if not ref_path:
            return ''
        cleaned = ref_path.strip()
        cleaned = cleaned.split('#', 1)[0].split('?', 1)[0]
        if not cleaned:
            return ''
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', cleaned):
            return ''
        if cleaned.startswith('/'):
            candidate = cleaned.lstrip('/')
        else:
            candidate = posixpath.normpath(
                posixpath.join(posixpath.dirname(base_doc_path), cleaned))
        if candidate.startswith('../'):
            return ''
        return candidate

    def infer_component(source_path):
        parts = [part for part in source_path.split('/') if part]
        preferred = {
            'pipelines',
            'katib',
            'notebooks',
            'training-operator',
            'kserve',
            'centraldashboard',
            'manifests',
            'sdk',
            'backend',
            'frontend',
        }
        for part in parts:
            if part in preferred:
                return part
        return parts[0] if parts else 'unknown'

    visual_index = {}
    with open(structured_visual_artifacts.path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                visual_index[record['source_path']] = record

    with open(structured_text_chunks.path, 'r', encoding='utf-8') as inp, open(
            structured_knowledge_artifacts.path, 'w',
            encoding='utf-8') as out:
        for line in inp:
            if not line.strip():
                continue
            text_record = json.loads(line)
            resolved_visual_refs = []
            visual_context = []
            for image_ref in text_record.get('image_references', []):
                normalized_ref = normalize_repo_path(text_record['source_path'],
                                                    image_ref)
                if not normalized_ref:
                    continue
                resolved_visual_refs.append(normalized_ref)
                visual_record = visual_index.get(normalized_ref)
                if visual_record:
                    visual_context.append(visual_record)

            deduped_visual_context = []
            seen_paths = set()
            for visual in visual_context:
                path = visual.get('source_path')
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                deduped_visual_context.append(visual)

            artifact = {
                'chunk_id': text_record.get('chunk_id'),
                'source_path': text_record['source_path'],
                'header_path': text_record['header_path'],
                'text_content': text_record['text_content'],
                'code_blocks': text_record.get('code_blocks', []),
                'image_references': text_record.get('image_references', []),
                'resolved_visual_references': sorted(set(resolved_visual_refs)),
                'visual_context': deduped_visual_context,
                'commit_hash': provenance['commit_hash'],
                'git_ref': provenance.get('git_ref'),
                'snapshot_root': provenance.get('snapshot_root', 'repo'),
                'source_url': provenance['repo_url'],
                'ingested_at_utc': provenance.get('ingested_at_utc'),
                'component': infer_component(text_record['source_path']),
            }
            out.write(json.dumps(artifact) + '\n')


@dsl.component(kfp_package_path=_KFP_PACKAGE_PATH)
def validate_ingestion_output(
    structured_knowledge_artifacts: Input[Dataset],
    source_volume: Input[Artifact],
    validated_knowledge_bank: Output[Dataset],
    validation_report: Output[Artifact],
):
    """Stage 4: Apply structural, navigational, and semantic guardrails.

    Guardrails:
    - Structural integrity: required-field checks (drops invalid records)
    - Navigational integrity: internal Markdown link existence checks
    - Semantic quality: remove stubs/noise and tag low-stability content

    Outputs:
    - validated_knowledge_bank: validated/tagged JSONL artifacts
    - validation_report: HTML summary with metrics and sample findings
    """
    import json
    import os
    import posixpath
    import re

    repo_root = os.path.join(source_volume.path, 'repo')

    required_fields = (
        'chunk_id',
        'source_path',
        'text_content',
        'commit_hash',
        'source_url',
        'component',
    )

    todo_markers = ('todo', 'wip', 'coming soon')
    stability_markers = {
        'deprecated': 'low_stability:deprecated',
        'alpha': 'low_stability:alpha',
        'experimental': 'low_stability:experimental',
    }
    link_pattern = re.compile(r'\[[^\]]+\]\(([^)]+)\)')

    def normalize_repo_path(base_doc_path, ref_path):
        cleaned = (ref_path or '').strip()
        cleaned = cleaned.split('#', 1)[0].split('?', 1)[0]
        if not cleaned:
            return ''
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', cleaned):
            return ''
        if cleaned.startswith('/'):
            return cleaned.lstrip('/')
        normalized = posixpath.normpath(
            posixpath.join(posixpath.dirname(base_doc_path), cleaned))
        if normalized.startswith('../'):
            return ''
        return normalized

    stats = {
        'total': 0,
        'validated': 0,
        'dropped_schema': 0,
        'dropped_stub': 0,
        'artifacts_with_broken_links': 0,
        'total_broken_links': 0,
        'artifacts_with_stability_warnings': 0,
    }
    schema_drop_examples = []
    stub_drop_examples = []
    broken_link_examples = []

    with open(structured_knowledge_artifacts.path, 'r', encoding='utf-8') as inp, open(
            validated_knowledge_bank.path, 'w',
            encoding='utf-8') as out:
        for line in inp:
            if not line.strip():
                continue
            stats['total'] += 1
            record = json.loads(line)

            missing = [field for field in required_fields if not record.get(field)]
            if missing:
                stats['dropped_schema'] += 1
                if len(schema_drop_examples) < 10:
                    schema_drop_examples.append({
                        'chunk_id': record.get('chunk_id'),
                        'source_path': record.get('source_path'),
                        'missing_fields': missing,
                    })
                continue

            text = record.get('text_content', '')
            lowered = text.lower()
            if len(text.strip()) < 50 or any(marker in lowered
                                             for marker in todo_markers):
                stats['dropped_stub'] += 1
                if len(stub_drop_examples) < 10:
                    stub_drop_examples.append({
                        'chunk_id': record.get('chunk_id'),
                        'source_path': record.get('source_path'),
                        'reason': 'stub_or_todo',
                    })
                continue

            quality_tags = []
            validation = {
                'schema_ok': True,
                'navigational_integrity': 'ok',
                'semantic_quality': 'ok',
                'broken_links': [],
            }

            for marker, tag in stability_markers.items():
                if marker in lowered:
                    quality_tags.append(tag)

            if quality_tags:
                stats['artifacts_with_stability_warnings'] += 1

            source_path = record.get('source_path', '')
            broken_links = []
            for raw_link in link_pattern.findall(text):
                normalized = normalize_repo_path(source_path, raw_link)
                if not normalized:
                    continue
                full_target = os.path.join(repo_root, normalized)
                if not os.path.exists(full_target):
                    broken_links.append({
                        'raw_link': raw_link,
                        'resolved_path': normalized,
                    })

            if broken_links:
                validation['navigational_integrity'] = 'broken_links_detected'
                validation['broken_links'] = broken_links
                quality_tags.append('needs_link_check')
                stats['artifacts_with_broken_links'] += 1
                stats['total_broken_links'] += len(broken_links)
                if len(broken_link_examples) < 10:
                    broken_link_examples.append({
                        'chunk_id': record.get('chunk_id'),
                        'source_path': source_path,
                        'broken_links': broken_links,
                    })

            record['quality_tags'] = sorted(set(quality_tags))
            record['validation'] = validation

            stats['validated'] += 1
            out.write(json.dumps(record) + '\n')

    validity_ratio = 0 if stats['total'] == 0 else round(
        stats['validated'] / stats['total'], 4)
    report = f"""<html><body>
<h1>Validation Report</h1>
<h2>Summary</h2>
<ul>
  <li>Total artifacts: {stats['total']}</li>
  <li>Validated artifacts: {stats['validated']}</li>
  <li>Dropped by schema: {stats['dropped_schema']}</li>
  <li>Dropped as stub/noise: {stats['dropped_stub']}</li>
  <li>Artifacts with broken links: {stats['artifacts_with_broken_links']}</li>
  <li>Total broken links: {stats['total_broken_links']}</li>
  <li>Artifacts with stability warnings: {stats['artifacts_with_stability_warnings']}</li>
  <li>Validity ratio: {validity_ratio}</li>
</ul>
<h2>Sample dropped records (schema)</h2>
<pre>{json.dumps(schema_drop_examples, indent=2)}</pre>
<h2>Sample dropped records (stub/noise)</h2>
<pre>{json.dumps(stub_drop_examples, indent=2)}</pre>
<h2>Sample broken link findings</h2>
<pre>{json.dumps(broken_link_examples, indent=2)}</pre>
</body></html>"""
    with open(validation_report.path, 'w', encoding='utf-8') as f:
        f.write(report)


@dsl.component(kfp_package_path=_KFP_PACKAGE_PATH)
def persist_and_optimize(
    validated_knowledge_bank: Input[Dataset],
    retrieval_ready_records: Output[Dataset],
):
    """Stage 5: Build retrieval-ready records with idempotent persistence keys.

    Behavior:
    - Flattens contextual text (header + body + visual context)
    - Emits deterministic record_id/source_key hashes
    - Emits an upsert/delete_scope contract for downstream vector stores
    """
    import hashlib
    import json

    def hash_text(value):
        return hashlib.sha256(value.encode('utf-8')).hexdigest()

    seen_record_ids = set()
    with open(validated_knowledge_bank.path, 'r', encoding='utf-8') as inp, open(
            retrieval_ready_records.path, 'w',
            encoding='utf-8') as out:
        for line in inp:
            if not line.strip():
                continue
            record = json.loads(line)
            header_path = record.get('header_path', [])
            text_content = record.get('text_content', '')
            visual_context = record.get('visual_context', [])

            retrieval_text = '\n\n'.join([
                f"header_path: {' > '.join(header_path) if header_path else '(root)'}",
                f"text_content:\n{text_content}",
                f"visual_context:\n{json.dumps(visual_context, sort_keys=True)}",
            ]).strip()

            chunk_id = record.get('chunk_id', '')
            source_path = record.get('source_path', '')
            source_url = record.get('source_url', '')
            commit_hash = record.get('commit_hash', '')
            component = record.get('component', 'unknown')

            record_id = hash_text(
                f'{source_url}|{commit_hash}|{source_path}|{chunk_id}')
            if record_id in seen_record_ids:
                continue
            seen_record_ids.add(record_id)

            source_key = hash_text(f'{source_url}|{source_path}|{chunk_id}')
            quality_tags = sorted(set(record.get('quality_tags', [])))
            validation = record.get('validation', {})

            metadata_payload = {
                'record_id': record_id,
                'source_key': source_key,
                'source_url': source_url,
                'source_path': source_path,
                'chunk_id': chunk_id,
                'component': component,
                'commit_hash': commit_hash,
                'git_ref': record.get('git_ref'),
                'header_path': header_path,
                'quality_tags': quality_tags,
                'has_broken_links': bool(validation.get('broken_links')),
                'stability_tags': [
                    tag for tag in quality_tags if tag.startswith('low_stability:')
                ],
            }

            output_record = {
                'operation': 'upsert',
                'record_id': record_id,
                'delete_scope': {
                    'source_key': source_key,
                },
                'source_path': source_path,
                'commit_hash': commit_hash,
                'component': component,
                'retrieval_text': retrieval_text,
                'metadata': metadata_payload,
            }
            out.write(json.dumps(output_record) + '\n')


@dsl.pipeline(
    name='docs-agent-ingestion-backbone',
    description='Deterministic ingestion backbone for docs + diagrams.',
)
def docs_agent_ingestion_backbone_pipeline(
    repo_url: str = 'https://github.com/kubeflow/pipelines.git',
    git_ref: str = 'master',
):
    """End-to-end deterministic ingestion DAG for docs + diagram knowledge."""
    acquisition = fetch_sources(repo_url=repo_url, git_ref=git_ref)
    manifest = filter_manifest(source_volume=acquisition.outputs['source_volume'])

    text_track = extract_text_docs(
        source_volume=acquisition.outputs['source_volume'],
        text_file_manifest=manifest.outputs['text_file_manifest'],
    )
    visual_track = process_visual_artifacts(
        source_volume=acquisition.outputs['source_volume'],
        visual_file_manifest=manifest.outputs['visual_file_manifest'])

    integration = structure_platform_knowledge(
        structured_text_chunks=text_track.outputs['structured_text_chunks'],
        structured_visual_artifacts=visual_track.outputs[
            'structured_visual_artifacts'],
        provenance_metadata=acquisition.outputs['provenance_metadata'],
    )
    validation = validate_ingestion_output(
        structured_knowledge_artifacts=integration.outputs[
            'structured_knowledge_artifacts'],
        source_volume=acquisition.outputs['source_volume'])
    persist_and_optimize(
        validated_knowledge_bank=validation.outputs['validated_knowledge_bank'])


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=docs_agent_ingestion_backbone_pipeline,
        package_path=__file__ + '.yaml',
    )
