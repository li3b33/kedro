from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_summer_data, train_classification_models

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_summer_data,
            inputs="summer_with_medal_flag",
            outputs="classification_split_data",
            name="prepare_summer_data_node"
        ),
        node(
            func=train_classification_models,
            inputs="classification_split_data",
            outputs="classification_results",
            name="train_classification_models_node"
        )
    ])
