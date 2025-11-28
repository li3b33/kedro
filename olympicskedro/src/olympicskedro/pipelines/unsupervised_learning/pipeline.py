from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_clustering_data,
    run_complete_unsupervised_analysis
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=run_complete_unsupervised_analysis,
            inputs=["summer_with_medal_flag", "dictionary_scaled"],
            outputs={
                "clustering_results": "clustering_results",
                "metrics": "clustering_metrics", 
                "pca_info": "pca_results",
                "models": "unsupervised_models",
                "wcss": "wcss_data"
            },
            name="run_unsupervised_analysis_node"
        )
    ])