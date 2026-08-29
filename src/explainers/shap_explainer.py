import shap
import pandas as pd

def create_explainer(model):
    return shap.TreeExplainer(model)


def explain_instance(explainer, instance, instance_id):
    explanation = explainer(instance)

    shap_df = pd.DataFrame({
    'instance_id': [instance_id] * len(instance.columns),
    'feature': instance.columns,
    'feature_value': instance.iloc[0].astype(str).values,
    'importance': explanation.values[0]
    })

    return shap_df