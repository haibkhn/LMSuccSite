# This is for without phi/psi

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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
    # Extract and normalize SASA
    sasa = df['sasa'].values.reshape(-1, 1)
    
    # Convert secondary structure one-hot encoding
    ss = np.column_stack((df['E'], df['H'], df['L']))
    
    return sasa, ss

def create_sequence_model():
    """Create CNN model for sequence data"""
    seq_input = tf.keras.layers.Input(shape=(33,))
    x = tf.keras.layers.Embedding(21, 21)(seq_input)
    x = tf.keras.layers.Reshape((33, 21, 1))(x)
    x = tf.keras.layers.Conv2D(32, kernel_size=(17, 3), activation='relu', padding='valid')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    seq_features = tf.keras.layers.Dense(32, activation='relu')(x)
    
    return tf.keras.Model(inputs=seq_input, outputs=seq_features)

def create_structure_model():
    """Create model for structural features"""
    # SASA input
    sasa_input = tf.keras.layers.Input(shape=(1,))
    sasa_features = tf.keras.layers.Dense(8, activation='relu')(sasa_input)
    
    # Secondary structure input
    ss_input = tf.keras.layers.Input(shape=(3,))
    ss_features = tf.keras.layers.Dense(8, activation='relu')(ss_input)
    
    # Combine structural features
    combined = tf.keras.layers.Concatenate()([sasa_features, ss_features])
    struct_features = tf.keras.layers.Dense(32, activation='relu')(combined)
    
    return tf.keras.Model(
        inputs=[sasa_input, ss_input],
        outputs=struct_features
    )

def create_ensemble_model():
    """Create ensemble model combining sequence and structure"""
    # Sequence branch
    seq_input = tf.keras.layers.Input(shape=(33,))
    seq_model = create_sequence_model()
    seq_features = seq_model(seq_input)
    
    # Structure branch
    sasa_input = tf.keras.layers.Input(shape=(1,))
    ss_input = tf.keras.layers.Input(shape=(3,))
    
    struct_model = create_structure_model()
    struct_features = struct_model([sasa_input, ss_input])
    
    # Combine features
    combined = tf.keras.layers.Concatenate()([seq_features, struct_features])
    
    # Final dense layers
    x = tf.keras.layers.Dense(64, activation='relu')(combined)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(
        inputs=[seq_input, sasa_input, ss_input],
        outputs=output
    )

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
        print(f"\nFold {fold}/10")
        
        # Split data
        train_data = [
            X_train_seq[train_idx],
            X_train_sasa[train_idx],
            X_train_ss[train_idx]
        ]
        val_data = [
            X_train_seq[val_idx],
            X_train_sasa[val_idx],
            X_train_ss[val_idx]
        ]
        
        # Create callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        )
        
        # Create and compile model
        model = create_ensemble_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        print("Training model...")
        class_weights = {0: 1.5, 1: 1}
        history = model.fit(
            train_data, y_train[train_idx],
            batch_size=32,
            epochs=50,
            validation_data=(val_data, y_train[val_idx]),
            callbacks=[early_stopping],
            verbose=1,
            class_weight=class_weights
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
        y_pred = model.predict(val_data)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(y_train[val_idx], y_pred_binary)
        metrics['acc'].append(accuracy_score(y_train[val_idx], y_pred_binary))
        metrics['mcc'].append(matthews_corrcoef(y_train[val_idx], y_pred_binary))
        metrics['sn'].append(cm[1][1]/(cm[1][1]+cm[1][0]))
        metrics['sp'].append(cm[0][0]/(cm[0][0]+cm[0][1]))
        
        # Predict on test set
        test_pred = model.predict([X_test_seq, X_test_sasa, X_test_ss])
        test_predictions.append(test_pred)
        
        print(f"\nFold {fold} Results:")
        print(f"Accuracy: {metrics['acc'][-1]:.4f}")
        print(f"MCC: {metrics['mcc'][-1]:.4f}")
        print(f"Sensitivity: {metrics['sn'][-1]:.4f}")
        print(f"Specificity: {metrics['sp'][-1]:.4f}")
    
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
    
    return model

if __name__ == "__main__":
    model = train_and_evaluate()
