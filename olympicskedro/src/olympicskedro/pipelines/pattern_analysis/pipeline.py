from kedro.pipeline import Pipeline, node, pipeline
from .nodes import analyze_cluster_patterns

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=analyze_cluster_patterns,
            inputs=["summer_with_medal_flag", "clustering_results", "pca_results"],
            outputs="pattern_analysis_results",
            name="analyze_cluster_patterns_node"
        )
    ])