from sklearn.discriminant_analysis import StandardScaler
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras.layers import (
    Conv2D, Dense, MaxPooling2D, Input, Flatten, Dropout, 
    Lambda, LeakyReLU, Embedding, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def prepare_sequence_data(df):
    """Convert sequences to integer encoding"""
    alphabet = 'ARNDCQEGHILKMFPSTWYV-'
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    
    sequences = df['sequence'].values
    encodings = []
    
    for seq in sequences:
        try:
            integer_encoded = [char_to_int[char] for char in seq]
            encodings.append(integer_encoded)
        except Exception as e:
            print(f"Error processing sequence: {e}")
            continue
    
    return np.array(encodings)

def prepare_structure_data(df):
    """Prepare structural features"""
    sasa = df['sasa'].values.reshape(-1, 1)
    
    # Preprocess sasa
    scaler = StandardScaler()
    sasa_scaled = scaler.fit_transform(sasa)
    sasa = sasa_scaled
    
    ss = np.column_stack((df['E'], df['H'], df['L']))
    return sasa, ss

def create_embedding_model():
    """Create CNN model for sequence data with modified architecture"""
    seq_input = Input(shape=(33,))
    x = Embedding(256, 33, input_length=33)(seq_input)
    x = Lambda(lambda x: tf.expand_dims(x, 3))(x)
    
    # Reduce dropout rates (current 0.6 might be too high)
    x = Conv2D(64, kernel_size=(17, 3), activation='relu', padding='valid')(x)
    x = Dropout(0.3)(x)  # Reduced from 0.6
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.3)(x)  # Reduced from 0.6
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(768, activation='relu')(x)
    x = Dropout(0.3)(x)  # Reduced from 0.5
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)  # Reduced from 0.5
    features = Dense(128, activation='relu')(x)
    
    return Model(inputs=seq_input, outputs=features)

def create_structure_model():
    """Create model for structural features"""
    sasa_input = Input(shape=(1,))
    ss_input = Input(shape=(3,))
    
    # SASA branch
    sasa_features = Dense(8, activation='relu')(sasa_input)
    
    # Secondary structure branch
    ss_features = Dense(8, activation='relu')(ss_input)
    
    # Combine structural features
    combined = concatenate([sasa_features, ss_features])
    features = Dense(32, activation='relu')(combined)
    
    return Model(inputs=[sasa_input, ss_input], outputs=features)

def create_combined_model(embedding_model, structure_model):
    """Create combined model with sequence and structure"""
    # Get outputs from base models
    combined = concatenate([
        embedding_model.output,
        structure_model.output
    ])
    
    # Add final layers
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    # Create combined model
    combined_model = Model(
        inputs=[
            embedding_model.input,
            *structure_model.inputs
        ],
        outputs=output
    )
    
    return combined_model

def train_and_evaluate():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv("./processed_data_train.csv")
    test_df = pd.read_csv("./processed_data_test.csv")
    
    # Prepare sequence data
    print("Preparing sequence data...")
    X_train_seq = prepare_sequence_data(train_df)
    X_test_seq = prepare_sequence_data(test_df)
    
    # Prepare structure data
    print("Preparing structure data...")
    X_train_sasa, X_train_ss = prepare_structure_data(train_df)
    X_test_sasa, X_test_ss = prepare_structure_data(test_df)
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    # Initialize cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize metrics storage
    metrics = {'acc': [], 'mcc': [], 'sn': [], 'sp': []}
    test_predictions = []
    
    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_seq), 1):
        print(f"\nFold {fold}/5")
        
        # Split data
        X_train_seq_fold = X_train_seq[train_idx]
        X_train_sasa_fold = X_train_sasa[train_idx]
        X_train_ss_fold = X_train_ss[train_idx]
        y_train_fold = y_train[train_idx]
        
        X_val_seq_fold = X_train_seq[val_idx]
        X_val_sasa_fold = X_train_sasa[val_idx]
        X_val_ss_fold = X_train_ss[val_idx]
        y_val_fold = y_train[val_idx]
        
        # Create base models
        embedding_model = create_embedding_model()
        structure_model = create_structure_model()
        
        # Create and compile combined model
        combined_model = create_combined_model(embedding_model, structure_model)
        combined_model.compile(
            optimizer=Adam(learning_rate=5e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Create callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True
        )
        
        # Train model
        print("Training model...")
        class_weights = {0: 1.3, 1: 1}
        history = combined_model.fit(
            [X_train_seq_fold, X_train_sasa_fold, X_train_ss_fold],
            y_train_fold,
            validation_data=(
                [X_val_seq_fold, X_val_sasa_fold, X_val_ss_fold],
                y_val_fold
            ),
            batch_size=64,
            epochs=50,
            callbacks=[early_stopping],
            verbose=1,
            class_weight=class_weights,
            
        )
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Model Accuracy - Fold {fold}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        # plt.show()
        
        # Evaluate on validation set
        y_pred = combined_model.predict(
            [X_val_seq_fold, X_val_sasa_fold, X_val_ss_fold]
        )
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(y_val_fold, y_pred_binary)
        metrics['acc'].append(accuracy_score(y_val_fold, y_pred_binary))
        metrics['mcc'].append(matthews_corrcoef(y_val_fold, y_pred_binary))
        metrics['sn'].append(cm[1][1]/(cm[1][1]+cm[1][0]))
        metrics['sp'].append(cm[0][0]/(cm[0][0]+cm[0][1]))
        
        # Predict on test set
        test_pred = combined_model.predict(
            [X_test_seq, X_test_sasa, X_test_ss]
        )
        test_predictions.append(test_pred)
        
        print(f"\nFold {fold} Results:")
        print(f"Accuracy: {metrics['acc'][-1]:.4f}")
        print(f"MCC: {metrics['mcc'][-1]:.4f}")
        print(f"Sensitivity: {metrics['sn'][-1]:.4f}")
        print(f"Specificity: {metrics['sp'][-1]:.4f}")
        print("Confusion Matrix:")
        print(cm)
    
    # Print average cross-validation results
    print("\nAverage Cross-validation Results:")
    for metric in metrics:
        print(f"{metric.upper()}: {np.mean(metrics[metric]):.4f} ± {np.std(metrics[metric]):.4f}")
    
    # Ensemble predictions on test set
    test_pred_avg = np.mean(test_predictions, axis=0)
    test_pred_binary = (test_pred_avg > 0.5).astype(int)
    
    # Calculate final test metrics
    cm_test = confusion_matrix(y_test, test_pred_binary)
    print("\nFinal Test Set Results:")
    print(f"Accuracy: {accuracy_score(y_test, test_pred_binary):.4f}")
    print(f"MCC: {matthews_corrcoef(y_test, test_pred_binary):.4f}")
    print(f"Sensitivity: {cm_test[1][1]/(cm_test[1][1]+cm_test[1][0]):.4f}")
    print(f"Specificity: {cm_test[0][0]/(cm_test[0][0]+cm_test[0][1]):.4f}")
    print("Confusion Matrix:")
    print(cm_test)
    print("AUC: ", roc_auc_score(y_test, test_pred_avg))

if __name__ == "__main__":
    train_and_evaluate()