from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

def create_pipeline():
    
    onehot_cols = ['chest_pain_type', 'thalassemia', 'rest_ecg']
    
    numeric_cols = [
        'age',
        'resting_blood_pressure',
        'cholestoral',
        'Max_heart_rate',
        'oldpeak',
        'vessels_colored_by_flourosopy',
    ]

    preprocessor = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), onehot_cols),
        ('num', StandardScaler(), numeric_cols)
    ], remainder='passthrough')

    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', LogisticRegression(C=0.1,  penalty='l2', solver='lbfgs', max_iter=1000))
    ])

    return pipeline