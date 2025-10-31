from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_classification_summary, generate_regression_summary, combine_final_summary

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_classification_summary,
            inputs="classification_results",
            outputs="classification_summary",
            name="generate_classification_summary"
        ),
        node(
            func=generate_regression_summary,
            inputs="regression_results",
            outputs="regression_summary",
            name="generate_regression_summary"
        ),
        node(
            func=combine_final_summary,
            inputs=["classification_summary", "regression_summary"],
            outputs="final_model_summary",
            name="combine_final_summary"
        )
    ])
