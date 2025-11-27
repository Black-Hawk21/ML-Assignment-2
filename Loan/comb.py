import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Load and preprocess data
def preprocess_data(df):
    """Preprocess the loan default dataset"""
    # Separate features and target
    X = df.drop(['Default', 'LoanID'], axis=1)
    y = df['Default']
    
    # Encode categorical variables
    categorical_cols = ['Education', 'EmploymentType', 'MaritalStatus', 
                       'HasMortgage', 'HasDependents', 'LoanPurpose', 'HasCoSigner']
    
    label_encoders = {}
    for col in categorical_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, X.shape[1]


# 2. ADABOOST-INSPIRED NEURAL NETWORK
class AdaBoostNN(nn.Module):
    """
    Neural network mimicking AdaBoost by using multiple weak learners
    with weighted combining and sequential training emphasis
    """
    def __init__(self, input_dim, n_estimators=5):
        super(AdaBoostNN, self).__init__()
        self.n_estimators = n_estimators
        
        # Create multiple weak learners (shallow networks)
        self.weak_learners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Tanh()  # tanh for -1 to 1 range
            ) for _ in range(n_estimators)
        ])
        
        # Learnable weights for each estimator (mimicking alpha in AdaBoost)
        self.estimator_weights = nn.Parameter(
            torch.ones(n_estimators) / n_estimators
        )
    
    def forward(self, x):
        # Get predictions from all weak learners
        predictions = []
        for learner in self.weak_learners:
            pred = learner(x)
            predictions.append(pred)
        
        # Stack predictions
        stacked = torch.stack(predictions, dim=-1)  # shape: (batch, 1, n_estimators)
        
        # Apply softmax to weights for proper weighting
        normalized_weights = torch.softmax(self.estimator_weights, dim=0)
        
        # Weighted combination
        weighted_sum = torch.sum(stacked * normalized_weights, dim=-1)
        
        # Final activation
        output = torch.sigmoid(weighted_sum)
        
        return output


# 3. CATBOOST-INSPIRED NEURAL NETWORK
class CatBoostNN(nn.Module):
    """
    Neural network mimicking CatBoost with:
    - Ordered boosting via sequential residual learning
    - Feature combinations through interaction layers
    - Gradient-based feature importance
    """
    def __init__(self, input_dim, n_stages=4):
        super(CatBoostNN, self).__init__()
        self.n_stages = n_stages
        self.input_dim = input_dim
        
        # Feature interaction layer (mimics CatBoost's categorical feature combinations)
        self.feature_interaction = nn.Linear(input_dim, input_dim * 2)
        self.interaction_bn = nn.BatchNorm1d(input_dim * 2)
        self.interaction_activation = nn.ReLU()
        
        # Feature attention for feature importance
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=1)
        )
        
        # Sequential boosting stages (ordered boosting concept)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim * 2 + 1, 64),  # +1 for previous stage output
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)  # Residual output (no activation)
            ) for _ in range(n_stages)
        ])
    
    def forward(self, x):
        # Apply feature attention (feature importance)
        attention_weights = self.feature_attention(x)
        weighted_features = x * attention_weights
        
        # Feature interaction
        interactions = self.feature_interaction(weighted_features)
        interactions = self.interaction_bn(interactions)
        interactions = self.interaction_activation(interactions)
        
        # Sequential boosting stages with residual learning
        stage_output = torch.zeros((x.size(0), 1), device=x.device)
        
        for stage in self.stages:
            # Concatenate original features and current prediction
            stage_input = torch.cat([interactions, stage_output], dim=1)
            # Predict residual
            residual = stage(stage_input)
            # Add residual to current prediction
            stage_output = stage_output + residual
        
        # Final activation
        output = torch.sigmoid(stage_output)
        
        return output


# 4. RESNET-INSPIRED NEURAL NETWORK
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
        
        # Residual connection
        out = out + identity
        
        return out


class ResNetNN(nn.Module):
    """
    ResNet-inspired architecture with residual connections
    Mimics gradient flow benefits similar to boosting
    """
    def __init__(self, input_dim, n_blocks=4):
        super(ResNetNN, self).__init__()
        
        # Initial projection to fixed dimension
        self.initial_dense = nn.Linear(input_dim, 128)
        self.initial_bn = nn.BatchNorm1d(128)
        self.initial_relu = nn.ReLU()
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResNetBlock(128, dropout_rate=0.3) for _ in range(n_blocks)
        ])
        
        # Final layers
        self.final_dense1 = nn.Linear(128, 64)
        self.final_relu = nn.ReLU()
        self.final_dropout = nn.Dropout(0.3)
        self.output_layer = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Initial projection
        x = self.initial_dense(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)
        
        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Final layers
        x = self.final_dense1(x)
        x = self.final_relu(x)
        x = self.final_dropout(x)
        x = self.output_layer(x)
        output = self.sigmoid(x)
        
        return output


# Training function
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001):
    """Train a PyTorch model"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


# Evaluation function
def evaluate_model(model, test_loader, y_test):
    """Evaluate model and return metrics"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_probabilities.extend(outputs.cpu().numpy())
            predictions = (outputs > 0.5).float()
            all_predictions.extend(predictions.cpu().numpy())
    
    y_pred = np.array(all_predictions).flatten()
    y_pred_proba = np.array(all_probabilities).flatten()
    y_true = y_test.numpy().flatten()
    
    # Calculate metrics
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\nAUC-ROC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    return auc_score


# Training and evaluation wrapper
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name, epochs=50, save_path='models'):
    """Train model and return evaluation metrics"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    
    # Split training data for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Move model to device
    model = model.to(device)
    
    # Train
    model, history = train_model(model, train_loader, val_loader, epochs=epochs)
    
    # Evaluate
    print(f"\n{model_name} Results:")
    auc_score = evaluate_model(model, test_loader, y_test)
    
    # Save the trained model
    import os
    os.makedirs(save_path, exist_ok=True)
    model_filename = f"{model_name.replace(' ', '_').replace('-', '_').lower()}.pth"
    model_filepath = os.path.join(save_path, model_filename)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model_name,
        'auc_score': auc_score,
        'history': history
    }, model_filepath)
    
    print(f"\n✓ Model saved to: {model_filepath}")
    
    return history, auc_score


# Main execution
def main(df):
    """Main function to train all models"""
    # Preprocess data
    X_train, X_test, y_train, y_test, input_dim = preprocess_data(df)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Input dimension: {input_dim}")
    
    # Dictionary to store results
    results = {}
    
    print("\n" + "="*60)
    print("STARTING MODEL TRAINING")
    print("="*60)
    
    # 2. AdaBoost NN
    model_adaboost = AdaBoostNN(input_dim, n_estimators=5)
    history_adaboost, auc_adaboost = train_and_evaluate(
        model_adaboost, X_train, X_test, y_train, y_test, 
        "AdaBoost-Inspired Neural Network", epochs=50
    )
    results['AdaBoost-NN'] = auc_adaboost
    
    # 3. CatBoost NN
    model_catboost = CatBoostNN(input_dim, n_stages=4)
    history_catboost, auc_catboost = train_and_evaluate(
        model_catboost, X_train, X_test, y_train, y_test, 
        "CatBoost-Inspired Neural Network", epochs=50
    )
    results['CatBoost-NN'] = auc_catboost
    
    # 4. ResNet NN
    model_resnet = ResNetNN(input_dim, n_blocks=4)
    history_resnet, auc_resnet = train_and_evaluate(
        model_resnet, X_train, X_test, y_train, y_test, 
        "ResNet-Inspired Neural Network", epochs=50
    )
    results['ResNet-NN'] = auc_resnet
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*60}")
    for model_name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name:30s}: AUC-ROC = {auc:.4f}")
    
    print(f"\n✓ All models saved in './models/' directory")
    
    return results


# Usage example:
df = pd.read_csv('./Loan_Default.csv')
results = main(df)