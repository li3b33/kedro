from kedro.pipeline import Pipeline, node
from .nodes import load_data, clean_data, scale_features, create_medal_flag, save_processed_data

def create_pipeline(**kwargs):
    """Definir el flujo de procesamiento de datos"""
    return Pipeline([
        node(
            func=load_data,
            inputs=None,
            outputs=["winter", "summer", "dictionary"],
            name="load_data_node"
        ),
        node(
            func=clean_data,
            inputs=["winter", "summer", "dictionary"],
            outputs=["winter_procesed", "summer_procesed", "dictionary_procesed"],
            name="clean_data_node"
        ),
        node(
            func=scale_features,
            inputs="dictionary_procesed",
            outputs="dictionary_scaled",
            name="scale_features_node"
        ),
        node(
            func=create_medal_flag,
            inputs="summer_procesed",
            outputs="summer_with_medal_flag",
            name="create_medal_flag_node"
        ),
        node(
            func=save_processed_data,
            inputs=["winter_procesed", "summer_with_medal_flag", "dictionary_scaled"],
            outputs=None,
            name="save_processed_data_node"
        )
    ])
