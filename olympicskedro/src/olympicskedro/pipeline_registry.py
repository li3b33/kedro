from typing import Dict
from kedro.pipeline import Pipeline
from olympicskedro.pipelines.reporting import pipeline as rp
from olympicskedro.pipelines.data_engineering import pipeline as de
from olympicskedro.pipelines.classification import pipeline as clf
from olympicskedro.pipelines.regression import pipeline as reg
from olympicskedro.pipelines.unsupervised_learning import pipeline as usl
from olympicskedro.pipelines.reporting_unsupervised import pipeline as rus
from olympicskedro.pipelines.integration import pipeline as intp
from olympicskedro.pipelines.pattern_analysis import pipeline as pa
from olympicskedro.pipelines.innovation import pipeline as inn  

def register_pipelines() -> Dict[str, Pipeline]:
    # ============ PIPELINES EXISTENTES ============
    data_engineering_pipeline = de.create_pipeline()
    classification_pipeline = clf.create_pipeline()
    regression_pipeline = reg.create_pipeline()
    reporting_pipeline = rp.create_pipeline()
    unsupervised_pipeline = usl.create_pipeline()
    reporting_unsupervised_pipeline = rus.create_pipeline()
    integration_pipeline = intp.create_pipeline()
    pattern_analysis_pipeline = pa.create_pipeline()
    
    innovation_pipeline = inn.create_pipeline()  

    # ============ PIPELINE COMPLETO ============
    full_pipeline = (
        data_engineering_pipeline + 
        classification_pipeline + 
        regression_pipeline + 
        unsupervised_pipeline + 
        integration_pipeline + 
        pattern_analysis_pipeline + 
        innovation_pipeline +  
        reporting_pipeline + 
        reporting_unsupervised_pipeline
    )

    # ============ PIPELINE SIN INNOVACIÓN (para desarrollo rápido) ============
    pipeline_without_innovation = (
        data_engineering_pipeline + 
        classification_pipeline + 
        regression_pipeline + 
        unsupervised_pipeline + 
        integration_pipeline + 
        pattern_analysis_pipeline + 
        reporting_pipeline + 
        reporting_unsupervised_pipeline
    )

    return {
        # ============ PIPELINE POR DEFECTO (ejecuta TODO) ============
        "__default__": full_pipeline,
        
        # ============ PIPELINES INDIVIDUALES ============
        "data_engineering": data_engineering_pipeline,
        "classification": classification_pipeline,
        "regression": regression_pipeline,
        "unsupervised": unsupervised_pipeline,
        "integration": integration_pipeline,
        "pattern_analysis": pattern_analysis_pipeline,
        "innovation": innovation_pipeline,  
        "reporting": reporting_pipeline,
        "reporting_unsupervised": reporting_unsupervised_pipeline,
        
        # ============ PIPELINES COMBINADOS ============
        "supervised_learning": classification_pipeline + regression_pipeline,
        "ml_pipelines": classification_pipeline + regression_pipeline + unsupervised_pipeline,
        "analysis_pipelines": unsupervised_pipeline + pattern_analysis_pipeline + reporting_unsupervised_pipeline,
        "advanced_analysis": unsupervised_pipeline + pattern_analysis_pipeline + innovation_pipeline,  
        
        # ============ PIPELINES PARA DESARROLLO ============
        "quick_test": data_engineering_pipeline + classification_pipeline,
        "data_processing": data_engineering_pipeline + unsupervised_pipeline,
        "model_training": classification_pipeline + regression_pipeline,
        
        # ============ PIPELINE COMPLETO ============
        "full_pipeline": full_pipeline,
        
        # ============ PIPELINE SIN REPORTING (para desarrollo) ============
        "full_without_reporting": (
            data_engineering_pipeline + 
            classification_pipeline + 
            regression_pipeline + 
            unsupervised_pipeline + 
            integration_pipeline + 
            pattern_analysis_pipeline + 
            innovation_pipeline  
        ),
        
        # ============ PIPELINE SIN INNOVACIÓN (más rápido) ============
        "full_without_innovation": pipeline_without_innovation,
        
        # ============ PIPELINE SOLO ANÁLISIS ============
        "complete_analysis": (
            unsupervised_pipeline + 
            integration_pipeline + 
            pattern_analysis_pipeline + 
            innovation_pipeline +  
            reporting_unsupervised_pipeline
        ),
        
        # ============ PIPELINE SOLO MODELADO ============
        "complete_modeling": (
            classification_pipeline + 
            regression_pipeline + 
            integration_pipeline + 
            reporting_pipeline
        ),
        
        # ============ PIPELINE DEMOSTRACIÓN ============
        "demo_pipeline": (
            data_engineering_pipeline + 
            classification_pipeline + 
            unsupervised_pipeline + 
            innovation_pipeline  
        )
    }