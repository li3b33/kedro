from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_dictionary_data, train_regression_models

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_dictionary_data,
            inputs="dictionary_scaled",
            outputs="regression_split_data",
            name="prepare_dictionary_data_node"
        ),
        node(
            func=train_regression_models,
            inputs="regression_split_data",
            outputs="regression_results",
            name="train_regression_models_node"
        )
    ])
