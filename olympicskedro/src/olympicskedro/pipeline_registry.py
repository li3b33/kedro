from typing import Dict
from kedro.pipeline import Pipeline
from olympicskedro.pipelines.reporting import pipeline as rp
from olympicskedro.pipelines.data_engineering import pipeline as de
from olympicskedro.pipelines.classification import pipeline as clf
from olympicskedro.pipelines.regression import pipeline as reg

def register_pipelines() -> Dict[str, Pipeline]:
    data_engineering_pipeline = de.create_pipeline()
    classification_pipeline = clf.create_pipeline()
    regression_pipeline = reg.create_pipeline()
    repoting_pipeline = rp.create_pipeline()

    return {
        "__default__": data_engineering_pipeline + classification_pipeline + regression_pipeline,
        "data_engineering": data_engineering_pipeline,
        "classification": classification_pipeline,
        "regression": regression_pipeline,
        "reporting": repoting_pipeline,
    }
