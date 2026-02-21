from kfp import local
from samples.core.docs_agent_ingestion_backbone.docs_agent_ingestion_backbone import docs_agent_ingestion_backbone_pipeline

local.init(runner=local.SubprocessRunner(use_venv=False))
docs_agent_ingestion_backbone_pipeline(
    repo_url='file:///home/shreeharsh157/Desktop/pipelines',
    git_ref='master',
)
