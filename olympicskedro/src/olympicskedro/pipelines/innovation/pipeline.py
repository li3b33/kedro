from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_complete_innovation_pipeline

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=run_complete_innovation_pipeline,
            inputs=[
                "clustering_results",
                "pattern_analysis_results", 
                "classification_results",
                "regression_results",
                "integration_comparison"
            ],
            outputs="innovation_results",
            name="run_innovation_pipeline_node"
        )
    ])