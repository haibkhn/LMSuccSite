import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler

def prepare_data(df):
    """Prepare structural features for model"""
    print(f"\nPreparing data from DataFrame with shape: {df.shape}")
    
    features_list = []
    print("\nProcessing features...")
    
    for _, row in df.iterrows():
        try:
            # Get angles
            phi = np.array(eval(str(row['phi_small']))) 
            psi = np.array(eval(str(row['psi_small'])))
            
            # Combine all features
            features = np.concatenate([
                phi,  # 3 phi angles
                psi,  # 3 psi angles
                [row['sasa']],  # 1 SASA value
                [row['E'], row['H'], row['L']]  # 3 secondary structure values
            ])
            features_list.append(features)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    X = np.array(features_list)
    y = np.array(df['label'].values)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y

def create_model(input_shape):
    """Create a deep neural network model"""
    model = tf.keras.Sequential([
        # Input layer with normalization
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        
        # First dense layer
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.3),
        
        # Second dense layer
        tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.2),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Main code
print("Starting script...")

try:
    # 1. Load and prepare data
    print("Loading data...")
    train_df = pd.read_csv("./processed_data_train.csv")
    test_df = pd.read_csv("./processed_data_test.csv")
    
    # 2. Prepare features
    X_train, y_train = prepare_data(train_df)
    X_test, y_test = prepare_data(test_df)
    
    # 3. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Create and compile model
    print("\nCreating model...")
    model = create_model(X_train.shape[1])
    
    # Calculate class weights
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    class_weight = {0: 1., 1: n_neg/n_pos}
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    
    # 5. Train model with early stopping and learning rate reduction
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        restore_best_weights=True,
        mode='max'
    )
    
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=5,
        mode='max',
        min_lr=0.00001
    )
    
    print("\nStarting training...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=[early_stopping, lr_reducer],
        verbose=1
    )
    
    # 6. Evaluate model
    print("\nEvaluating model...")
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    
    print("\nTest Metrics:")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 7. Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print(cm)

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nScript completed.")