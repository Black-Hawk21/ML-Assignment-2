import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# MODEL DEFINITIONS (must match training script)
# ============================================================================

class BaselineNN(nn.Module):
    """Simple feedforward neural network for baseline"""
    def __init__(self, input_dim):
        super(BaselineNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class AdaBoostNN(nn.Module):
    """AdaBoost-inspired neural network"""
    def __init__(self, input_dim, n_estimators=5):
        super(AdaBoostNN, self).__init__()
        self.n_estimators = n_estimators
        
        self.weak_learners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Tanh()
            ) for _ in range(n_estimators)
        ])
        
        self.estimator_weights = nn.Parameter(
            torch.ones(n_estimators) / n_estimators
        )
    
    def forward(self, x):
        predictions = []
        for learner in self.weak_learners:
            pred = learner(x)
            predictions.append(pred)
        
        stacked = torch.stack(predictions, dim=-1)
        normalized_weights = torch.softmax(self.estimator_weights, dim=0)
        weighted_sum = torch.sum(stacked * normalized_weights, dim=-1)
        output = torch.sigmoid(weighted_sum)
        
        return output


class CatBoostNN(nn.Module):
    """CatBoost-inspired neural network"""
    def __init__(self, input_dim, n_stages=4):
        super(CatBoostNN, self).__init__()
        self.n_stages = n_stages
        self.input_dim = input_dim
        
        self.feature_interaction = nn.Linear(input_dim, input_dim * 2)
        self.interaction_bn = nn.BatchNorm1d(input_dim * 2)
        self.interaction_activation = nn.ReLU()
        
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=1)
        )
        
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim * 2 + 1, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ) for _ in range(n_stages)
        ])
    
    def forward(self, x):
        attention_weights = self.feature_attention(x)
        weighted_features = x * attention_weights
        
        interactions = self.feature_interaction(weighted_features)
        interactions = self.interaction_bn(interactions)
        interactions = self.interaction_activation(interactions)
        
        stage_output = torch.zeros((x.size(0), 1), device=x.device)
        
        for stage in self.stages:
            stage_input = torch.cat([interactions, stage_output], dim=1)
            residual = stage(stage_input)
            stage_output = stage_output + residual
        
        output = torch.sigmoid(stage_output)
        
        return output


class ResNetBlock(nn.Module):
    """Residual block for ResNet architecture"""
    def __init__(self, units, dropout_rate=0.3):
        super(ResNetBlock, self).__init__()
        
        self.dense1 = nn.Linear(units, units)
        self.bn1 = nn.BatchNorm1d(units)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.dense2 = nn.Linear(units, units)
        self.bn2 = nn.BatchNorm1d(units)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        identity = x
        
        out = self.dense1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.dense2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = out + identity
        
        return out


class ResNetNN(nn.Module):
    """ResNet-inspired neural network"""
    def __init__(self, input_dim, n_blocks=4):
        super(ResNetNN, self).__init__()
        
        self.initial_dense = nn.Linear(input_dim, 128)
        self.initial_bn = nn.BatchNorm1d(128)
        self.initial_relu = nn.ReLU()
        
        self.res_blocks = nn.ModuleList([
            ResNetBlock(128, dropout_rate=0.3) for _ in range(n_blocks)
        ])
        
        self.final_dense1 = nn.Linear(128, 64)
        self.final_relu = nn.ReLU()
        self.final_dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.initial_dense(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        x = self.final_dense1(x)
        x = self.final_relu(x)
        x = self.final_dropout(x)
        x = self.output_layer(x)
        output = self.sigmoid(x)
        
        return output


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_test_data(test_df, train_df=None, scaler=None, label_encoders=None):
    """
    Preprocess test data for prediction
    
    Args:
        test_df: Test dataframe
        train_df: Training dataframe (optional, for fitting encoders)
        scaler: Pre-fitted scaler (optional)
        label_encoders: Pre-fitted label encoders (optional)
    
    Returns:
        X_test_tensor: Preprocessed test data as PyTorch tensor
        loan_ids: Original LoanIDs for submission
        scaler: Fitted scaler
        label_encoders: Fitted label encoders
    """
    # Save LoanIDs for submission
    loan_ids = test_df['LoanID'].values
    
    # Drop LoanID
    X_test = test_df.drop(['LoanID'], axis=1, errors='ignore')
    
    # Categorical columns
    categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 
                       'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
    
    # If train_df is provided, fit encoders on training data
    if train_df is not None and label_encoders is None:
        label_encoders = {}
        X_train = train_df.drop(['Default', 'LoanID'], axis=1, errors='ignore')
        
        for col in categorical_cols:
            if col in X_train.columns:
                le = LabelEncoder()
                le.fit(X_train[col].astype(str))
                label_encoders[col] = le
    
    # Apply encoders to test data
    if label_encoders is not None:
        for col in categorical_cols:
            if col in X_test.columns and col in label_encoders:
                # Handle unseen categories
                le = label_encoders[col]
                X_test[col] = X_test[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                X_test[col] = le.transform(X_test[col])
    
    # If train_df is provided, fit scaler on training data
    if train_df is not None and scaler is None:
        X_train = train_df.drop(['Default', 'LoanID'], axis=1, errors='ignore')
        # Apply same encoding to training data
        for col in categorical_cols:
            if col in X_train.columns and col in label_encoders:
                le = label_encoders[col]
                X_train[col] = le.transform(X_train[col].astype(str))
        
        scaler = StandardScaler()
        scaler.fit(X_train)
    
    # Scale test data
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test.values
    
    # Convert to tensor
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    return X_test_tensor, loan_ids, scaler, label_encoders


# ============================================================================
# MODEL LOADING AND PREDICTION
# ============================================================================

def load_model(model_filepath, model_class, input_dim, **model_kwargs):
    """Load a saved model"""
    model = model_class(input_dim, **model_kwargs)
    checkpoint = torch.load(model_filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Loaded model: {checkpoint['model_name']}")
    print(f"  AUC-ROC Score: {checkpoint['auc_score']:.4f}")
    
    return model, checkpoint


def make_predictions(model, X_test_tensor, batch_size=32):
    """Make predictions on test data"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            batch = X_test_tensor[i:i+batch_size].to(device)
            outputs = model(batch)
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions).flatten()


# ============================================================================
# MAIN SUBMISSION FUNCTION
# ============================================================================

def create_submission(
    test_csv_path='test.csv',
    train_csv_path=None,
    model_path='models/catboost_inspired_neural_network.pth',
    model_type='catboost',
    submission_csv_path='submission.csv',
    probability_output=False
):
    """
    Create Kaggle-style submission file
    
    Args:
        test_csv_path: Path to test.csv file
        train_csv_path: Path to train.csv (needed for fitting encoders/scaler)
        model_path: Path to saved model file
        model_type: Type of model ('baseline', 'adaboost', 'catboost', 'resnet')
        submission_csv_path: Output path for submission.csv
        probability_output: If True, output probabilities; if False, output binary predictions
    """
    print("="*70)
    print("KAGGLE SUBMISSION GENERATOR - LOAN DEFAULT PREDICTION")
    print("="*70)
    
    # Load test data
    print(f"\n[1/5] Loading test data from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    print(f"      Test data shape: {test_df.shape}")
    
    # Load training data if provided (for fitting preprocessing)
    scaler = None
    label_encoders = None
    if train_csv_path is not None:
        print(f"\n[2/5] Loading training data from: {train_csv_path}")
        train_df = pd.read_csv(train_csv_path)
        print(f"      Training data shape: {train_df.shape}")
    else:
        print("\n[2/5] No training data provided - using test data statistics")
        train_df = None
    
    # Preprocess test data
    print("\n[3/5] Preprocessing test data...")
    X_test_tensor, loan_ids, scaler, label_encoders = preprocess_test_data(
        test_df, train_df, scaler, label_encoders
    )
    input_dim = X_test_tensor.shape[1]
    print(f"      Preprocessed shape: {X_test_tensor.shape}")
    print(f"      Input dimension: {input_dim}")
    
    # Load model
    print(f"\n[4/5] Loading model from: {model_path}")
    
    # Map model type to class and parameters
    model_configs = {
        'baseline': (BaselineNN, {}),
        'adaboost': (AdaBoostNN, {'n_estimators': 5}),
        'catboost': (CatBoostNN, {'n_stages': 4}),
        'resnet': (ResNetNN, {'n_blocks': 4})
    }
    
    if model_type.lower() not in model_configs:
        raise ValueError(f"Invalid model_type. Choose from: {list(model_configs.keys())}")
    
    model_class, model_kwargs = model_configs[model_type.lower()]
    model, checkpoint = load_model(model_path, model_class, input_dim, **model_kwargs)
    
    # Make predictions
    print(f"\n[5/5] Making predictions...")
    predictions_proba = make_predictions(model, X_test_tensor)
    
    if probability_output:
        predictions_final = predictions_proba
        print(f"      Generated probability predictions (range: {predictions_final.min():.4f} to {predictions_final.max():.4f})")
    else:
        predictions_final = (predictions_proba > 0.5).astype(int)
        print(f"      Generated binary predictions (0: {(predictions_final==0).sum()}, 1: {(predictions_final==1).sum()})")
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'LoanID': loan_ids,
        'Default': predictions_final
    })
    
    # Save submission file
    submission_df.to_csv(submission_csv_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Submission file created successfully!")
    print(f"  Output file: {submission_csv_path}")
    print(f"  Number of predictions: {len(submission_df)}")
    print(f"\nSubmission Preview:")
    print(submission_df.head(10))
    print(f"{'='*70}\n")
    
    return submission_df


# ============================================================================
# ENSEMBLE PREDICTION (BONUS)
# ============================================================================

def create_ensemble_submission(
    test_csv_path='test.csv',
    train_csv_path=None,
    model_paths=None,
    submission_csv_path='ensemble_submission.csv',
    probability_output=False
):
    """
    Create submission using ensemble of multiple models
    
    Args:
        test_csv_path: Path to test.csv file
        train_csv_path: Path to train.csv
        model_paths: List of tuples [(model_path, model_type), ...]
        submission_csv_path: Output path for submission.csv
        probability_output: If True, output probabilities; if False, output binary predictions
    """
    if model_paths is None:
        model_paths = [
            ('models/baseline_neural_network.pth', 'baseline'),
            ('models/adaboost_inspired_neural_network.pth', 'adaboost'),
            ('models/catboost_inspired_neural_network.pth', 'catboost'),
            ('models/resnet_inspired_neural_network.pth', 'resnet')
        ]
    
    print("="*70)
    print("ENSEMBLE SUBMISSION GENERATOR - LOAN DEFAULT PREDICTION")
    print("="*70)
    
    # Load and preprocess data
    print(f"\n[1/3] Loading and preprocessing data...")
    test_df = pd.read_csv(test_csv_path)
    train_df = pd.read_csv(train_csv_path) if train_csv_path else None
    
    X_test_tensor, loan_ids, _, _ = preprocess_test_data(test_df, train_df)
    input_dim = X_test_tensor.shape[1]
    
    # Collect predictions from all models
    print(f"\n[2/3] Making predictions with {len(model_paths)} models...")
    all_predictions = []
    
    model_configs = {
        'baseline': (BaselineNN, {}),
        'adaboost': (AdaBoostNN, {'n_estimators': 5}),
        'catboost': (CatBoostNN, {'n_stages': 4}),
        'resnet': (ResNetNN, {'n_blocks': 4})
    }
    
    for model_path, model_type in model_paths:
        print(f"\n  Loading {model_type}...")
        model_class, model_kwargs = model_configs[model_type.lower()]
        model, _ = load_model(model_path, model_class, input_dim, **model_kwargs)
        predictions = make_predictions(model, X_test_tensor)
        all_predictions.append(predictions)
    
    # Average predictions (soft voting)
    print(f"\n[3/3] Combining predictions...")
    ensemble_predictions_proba = np.mean(all_predictions, axis=0)
    
    if probability_output:
        predictions_final = ensemble_predictions_proba
    else:
        predictions_final = (ensemble_predictions_proba > 0.5).astype(int)
    
    # Create submission
    submission_df = pd.DataFrame({
        'LoanID': loan_ids,
        'Default': predictions_final
    })
    
    submission_df.to_csv(submission_csv_path, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Ensemble submission file created successfully!")
    print(f"  Output file: {submission_csv_path}")
    print(f"  Number of models: {len(model_paths)}")
    print(f"  Number of predictions: {len(submission_df)}")
    print(f"\nSubmission Preview:")
    print(submission_df.head(10))
    print(f"{'='*70}\n")
    
    return submission_df


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":

    # Example 1: Single model submission (CatBoost-NN)
    # submission = create_submission(
    #     test_csv_path='./test_updated.csv',
    #     train_csv_path='./Loan_default.csv',  # Needed for preprocessing
    #     model_path='./models/resnet_inspired_neural_network.pth',
    #     model_type='resnet',
    #     submission_csv_path='./submission_resnet.csv',
    #     probability_output=False  # Set True for probability outputs
    # )
    
    # Example 3: Ensemble submission (combines all models)
    ensemble_submission = create_ensemble_submission(
        test_csv_path='./test_updated.csv',
        train_csv_path='./Loan_default.csv',  # Needed for preprocessing
        submission_csv_path='./submission_ens.csv',
        probability_output=False
    )