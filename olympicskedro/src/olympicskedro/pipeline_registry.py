from typing import Dict
from kedro.pipeline import Pipeline
from olympicskedro.pipelines.data_engineering import pipeline as de

def register_pipelines() -> Dict[str, Pipeline]:
    data_engineering = de.create_pipeline()
    return {
        "__default__": data_engineering,
        "data_engineering": data_engineering,
    }