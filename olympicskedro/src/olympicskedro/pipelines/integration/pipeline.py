from kedro.pipeline import Pipeline, node, pipeline
from .nodes import integrate_clustering_features, train_integrated_model, compare_performance, save_integration_metrics

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=integrate_clustering_features,
            inputs=["summer_with_medal_flag", "clustering_results"],
            outputs="integrated_split_data",
            name="integrate_clustering_features_node"
        ),
        node(
            func=train_integrated_model,
            inputs="integrated_split_data",
            outputs="integrated_model_results",
            name="train_integrated_model_node"
        ),
        node(
            func=compare_performance,
            inputs=["classification_results", "integrated_model_results"],
            outputs="integration_comparison",
            name="compare_performance_node"
        ),
        node(
            func=save_integration_metrics,
            inputs="integration_comparison",
            outputs="integration_metrics",
            name="save_integration_metrics_node"
        )
    ])