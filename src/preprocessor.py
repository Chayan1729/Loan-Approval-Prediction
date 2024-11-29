from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, MinMaxScaler
from imblearn.pipeline import Pipeline

# Define the numeric columns to transform
Simple_Imputer_Standard_scaler_NUM_col = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

# Define the transformer for numeric columns (simple imputation + scaling)
trf1 = Pipeline(
    steps=[
        ("Simple Imputer", SimpleImputer(strategy="mean")),
        ("MinMax scaling", MinMaxScaler())
    ]
)

# Define columns for KNN imputation
KNN_Imputer_NUM_col = ["Credit_History"]

# Define the transformer for KNN imputation + scaling
trf3 = Pipeline(
    steps=[
        ("KNN Imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("MinMaxScaler", MinMaxScaler())
    ]
)

# Define columns for ordinal categorical encoding
Simple_Imputer_ordinal_CAT_col = ["Dependents", "Property_Area", "Education"]
custom_categories = [
    ["0", "1", "2", "3+"],                # Dependents
    ["Rural", "Semiurban", "Urban"],      # Property_Area
    ["Not Graduate", "Graduate"]          # Education
]

# Define the transformer for ordinal encoding
trf4 = Pipeline(
    steps=[
        ("Simple Imputer", SimpleImputer(strategy="most_frequent")),
        ("Ordinal Encoder", OrdinalEncoder(categories=custom_categories, handle_unknown="use_encoded_value", unknown_value=-1))
    ]
)

# Define columns for one-hot encoding
Simple_Imputer_One_Hot_Encoder_CAT_col = ["Gender", "Married", "Self_Employed"]

# Define the transformer for one-hot encoding
trf5 = Pipeline(
    steps=[
        ("Simple Imputer", SimpleImputer(strategy="most_frequent")),
        ("One Hot Encoder", OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore"))
    ]
)

# Define the final preprocessor as a ColumnTransformer
final_preprocessor = ColumnTransformer(
    transformers=[
        ("Log transform + Simple Imputer + MinMax Scaling", trf1, Simple_Imputer_Standard_scaler_NUM_col),
        ("KNN Imputer", trf3, KNN_Imputer_NUM_col),
        ("Simple imputer + Ordinal Encoder", trf4, Simple_Imputer_ordinal_CAT_col),
        ("Simple imputer + One Hot Encoder", trf5, Simple_Imputer_One_Hot_Encoder_CAT_col)
    ]
)
