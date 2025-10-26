from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    clean_dataset,
    truncate_gdp,
    fill_missing_gdp,
    scale_dictionary,
    add_medal_flag
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_dataset,
            inputs="winter",
            outputs="winter_procesed",
            name="clean_winter_node"
        ),
        node(
            func=clean_dataset,
            inputs="summer",
            outputs="summer_procesed",
            name="clean_summer_node"
        ),
        node(
            func=clean_dataset,
            inputs="dictionary",
            outputs="dictionary_procesed",
            name="clean_dictionary_node"
        ),
        node(
            func=truncate_gdp,
            inputs="dictionary_procesed",
            outputs="dictionary_truncated",
            name="truncate_gdp_node"
        ),
        node(
            func=fill_missing_gdp,
            inputs="dictionary_truncated",
            outputs="dictionary_filled",
            name="fill_missing_gdp_node"
        ),
        node(
            func=scale_dictionary,
            inputs="dictionary_filled",
            outputs="dictionary_scaled",
            name="scale_dictionary_node"
        ),
        node(
            func=add_medal_flag,
            inputs="summer_procesed",
            outputs="summer_with_medal_flag",
            name="add_medal_flag_node"
        )
    ])
