from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
        process_winter,
        process_summer,
        process_dictionary,
        scale_dictionary,
        add_medal_flag,
    )

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=process_winter,
            inputs="winter",
            outputs="winter_procesed",
            name="process_winter_node"
        ),
        node(
            func=process_summer,
            inputs="summer",
            outputs="summer_procesed",
            name="process_summer_node"
        ),
        node(
            func=process_dictionary,
            inputs="dictionary",
            outputs="dictionary_procesed",
            name="process_dictionary_node"
        ),
        node(
            func=scale_dictionary,
            inputs="dictionary_procesed",
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

