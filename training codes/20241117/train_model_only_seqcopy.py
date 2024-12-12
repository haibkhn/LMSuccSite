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

def create_cnn_model():
    """Create CNN model with embedding"""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(21, 21, input_length=33),  # Changed from 256 to 21 (vocab size)
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 3)),
        tf.keras.layers.Conv2D(32, kernel_size=(17, 3), 
                              activation='relu', 
                              kernel_initializer='he_normal', 
                              padding='VALID'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu', 
                            kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def plot_history(history):
    """Plot training history"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.legend()
    # plt.show()

def train_and_evaluate():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv("./processed_data_train.csv")
    test_df = pd.read_csv("./processed_data_test.csv")
    
    # Prepare sequence data
    print("Preparing sequence data...")
    X_train_full = prepare_sequence_data(train_df)
    X_test = prepare_sequence_data(test_df)
    
    y_train_full = train_df['label'].values
    y_test = test_df['label'].values
    
    print(f"Training data shape: {X_train_full.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Initialize cross-validation
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Initialize metrics storage
    avg_acc, avg_mcc, avg_sp, avg_sn = 0, 0, 0, 0
    fold = 1
    
    # Store all predictions for later ensemble
    test_predictions = []
    
    # Cross-validation loop
    for train_idx, val_idx in kfold.split(X_train_full, y_train_full):
        print(f"\nFold {fold}/10")
        
        # Split data
        X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
        
        # Create callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            mode='auto',
            restore_best_weights=True
        )
        
        # Create and compile model
        model = create_cnn_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        print("Training model...")
        history = model.fit(
            X_train, y_train,
            batch_size=256,
            epochs=100,
            verbose=1,
            callbacks=[early_stopping],
            validation_data=(X_val, y_val)
        )
        
        # Plot training history
        plot_history(history)
        
        # Evaluate on validation set
        y_pred = model.predict(X_val).reshape(y_val.shape[0],)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        cm = confusion_matrix(y_val, y_pred_binary)
        mcc = matthews_corrcoef(y_val, y_pred_binary)
        acc = accuracy_score(y_val, y_pred_binary)
        sn = cm[1][1]/(cm[1][1]+cm[1][0])  # Sensitivity
        sp = cm[0][0]/(cm[0][0]+cm[0][1])  # Specificity
        
        print(f"\nFold {fold} Results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"MCC: {mcc:.4f}")
        print(f"Sensitivity: {sn:.4f}")
        print(f"Specificity: {sp:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        # Update averages
        avg_acc += acc
        avg_mcc += mcc
        avg_sp += sp
        avg_sn += sn
        
        # Predict on test set
        test_pred = model.predict(X_test)
        test_predictions.append(test_pred)
        
        fold += 1
    
    # Calculate and print average metrics
    n_folds = 10
    print("\nAverage Cross-validation Results:")
    print(f"Accuracy: {avg_acc/n_folds:.4f}")
    print(f"MCC: {avg_mcc/n_folds:.4f}")
    print(f"Sensitivity: {avg_sn/n_folds:.4f}")
    print(f"Specificity: {avg_sp/n_folds:.4f}")
    
    # Ensemble predictions on test set
    test_pred_avg = np.mean(test_predictions, axis=0)
    test_pred_binary = (test_pred_avg > 0.5).astype(int)
    
    # Calculate final test metrics
    cm_test = confusion_matrix(y_test, test_pred_binary)
    mcc_test = matthews_corrcoef(y_test, test_pred_binary)
    acc_test = accuracy_score(y_test, test_pred_binary)
    sn_test = cm_test[1][1]/(cm_test[1][1]+cm_test[1][0])
    sp_test = cm_test[0][0]/(cm_test[0][0]+cm_test[0][1])
    
    print("\nFinal Test Set Results:")
    print(f"Accuracy: {acc_test:.4f}")
    print(f"MCC: {mcc_test:.4f}")
    print(f"Sensitivity: {sn_test:.4f}")
    print(f"Specificity: {sp_test:.4f}")
    print("Confusion Matrix:")
    print(cm_test)
    
    return model

if __name__ == "__main__":
    model = train_and_evaluate()