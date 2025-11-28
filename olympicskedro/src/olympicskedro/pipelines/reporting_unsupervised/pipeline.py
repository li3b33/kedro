from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    generate_clustering_summary,
    create_visualization_data,
    generate_unsupervised_report
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_clustering_summary,
            inputs=["clustering_results", "clustering_metrics"],
            outputs="clustering_summary",
            name="generate_clustering_summary_node"
        ),
        node(
            func=create_visualization_data,
            inputs=["clustering_results", "pca_results"],
            outputs="clustering_visualization_data",
            name="create_visualization_data_node"
        ),
        node(
            func=generate_unsupervised_report,
            inputs=["clustering_summary", "clustering_visualization_data"],
            outputs="unsupervised_summary",
            name="generate_unsupervised_report_node"
        )
    ])