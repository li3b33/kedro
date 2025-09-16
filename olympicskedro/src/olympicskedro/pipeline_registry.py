from typing import Dict
from kedro.pipeline import Pipeline
from olympicskedro.pipelines import data_engineering as de

def register_pipelines() -> Dict[str, Pipeline]:
    de_pipeline = de.create_pipeline()
    return {
        "__default__": de_pipeline,
        "de": de_pipeline,
    }