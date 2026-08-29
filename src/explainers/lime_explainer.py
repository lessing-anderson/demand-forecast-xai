import pandas as pd

from lime.lime_tabular import LimeTabularExplainer


def create_lime_explainer(
    X_train,
    categorical_cols=None,
    random_state=42
):
    """Create LIME explainer and preprocessing metadata."""

    X_lime = X_train.copy()
    categorical_cols = categorical_cols or []

    # Removi features que são constantes no dataset de treino (se usar o dataset inteiro não teria esse problema), pois o LIME não consegue lidar com features constantes... 
    # mas depois tive que recolocar para treinar igual o modelo original

    # Detect constant features in the current training dataset
    constant_cols = [
        col
        for col in X_lime.columns
        if X_lime[col].nunique(dropna=False) <= 1
    ]

    constant_values = {
        col: X_lime[col].iloc[0]
        for col in constant_cols
    }

    # Constant features do not need to be perturbed by LIME
    X_lime = X_lime.drop(columns=constant_cols)

    # Keep only categorical columns still available to LIME
    categorical_cols_lime = [
        col
        for col in categorical_cols
        if col in X_lime.columns
    ]

    categorical_features = []
    categorical_names = {}
    category_maps = {}

    # Encode categorical features for LIME
    for col in categorical_cols_lime:

        X_lime[col] = X_lime[col].astype('category')

        categories = X_lime[col].cat.categories.tolist()
        idx = X_lime.columns.get_loc(col)

        categorical_features.append(idx)
        categorical_names[idx] = categories
        category_maps[col] = categories

        X_lime[col] = X_lime[col].cat.codes

    explainer = LimeTabularExplainer(
        training_data=X_lime.values,
        feature_names=X_lime.columns.tolist(),
        categorical_features=categorical_features,
        categorical_names=categorical_names,
        mode='regression',
        discretize_continuous=False, ##Por causa de um erro ao usar o LIME com as variaveis continuas, desativei a discretizacao das variaveis continuas
        random_state=random_state
    )

    metadata = {
        'model_feature_names': X_train.columns.tolist(),
        'lime_feature_names': X_lime.columns.tolist(),
        'categorical_cols': categorical_cols_lime,
        'category_maps': category_maps,
        'constant_values': constant_values
    }

    return explainer, X_lime, metadata


def transform_instance_for_lime(
    instance,
    metadata
):
    """Transform one original observation to LIME representation."""

    instance_lime = instance.copy()

    # Remove features excluded from LIME
    instance_lime = instance_lime[
        metadata['lime_feature_names']
    ].copy()

    # Encode categorical values using training categories
    for col in metadata['categorical_cols']:

        categories = metadata['category_maps'][col]

        try:
            code = categories.index(instance_lime[col])
        except ValueError:
            raise ValueError(
                f"Value '{instance_lime[col]}' from column '{col}' "
                "was not present in the LIME training data."
            )

        instance_lime[col] = code

    return instance_lime.astype(float)


def create_predict_fn(
    model,
    metadata
):
    """Create prediction function compatible with LIME."""

    def predict_fn(X):

        X_df = pd.DataFrame(
            X,
            columns=metadata['lime_feature_names']
        )

        # Restore categorical features expected by LightGBM
        for col in metadata['categorical_cols']:

            categories = metadata['category_maps'][col]

            codes = (
                X_df[col]
                .round()
                .astype(int)
                .clip(0, len(categories) - 1)
            )

            X_df[col] = pd.Categorical.from_codes(
                codes,
                categories=categories
            )

        # Restore features that were constant in training
        for col, value in metadata['constant_values'].items():

            if isinstance(value, str):
                X_df[col] = pd.Categorical(
                    [value] * len(X_df),
                    categories=[value]
                )
            else:
                X_df[col] = value

        # Restore exact feature order used by the model
        X_df = X_df[
            metadata['model_feature_names']
        ]

        return model.predict(X_df)

    return predict_fn


def explain_instance(
    explainer,
    predict_fn,
    instance,
    instance_lime,
    instance_id,
    num_samples=5000,
    num_features=None
):
    """Generate a LIME explanation for one observation."""

    if num_features is None:
        num_features = len(instance_lime)

    explanation = explainer.explain_instance(
        data_row=instance_lime.values,
        predict_fn=predict_fn,
        num_features=num_features,
        num_samples=num_samples
    )

    lime_weights = dict(explanation.as_list())

    result = pd.DataFrame({
        'instance_id': instance_id,
        'feature': instance_lime.index,
        'feature_value': [
            instance[feature]
            for feature in instance_lime.index
        ],
        'importance': [
            lime_weights.get(feature, 0.0)
            for feature in instance_lime.index
        ]
    })

    result = result.sort_values(
        'importance',
        key=abs,
        ascending=False
    ).reset_index(drop=True)

    return explanation, result