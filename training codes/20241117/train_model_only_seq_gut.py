import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

def prepare_sequence_data(df):
    """Prepare sequence data"""
    # define universe of possible input values
    alphabet = 'ARNDCQEGHILKMFPSTWYV-'
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    
    sequences = df['sequence'].values
    encodings = []
    
    for seq in sequences:
        try:
            integer_encoded = [char_to_int[char] for char in seq]
            encodings.append(integer_encoded)
        except Exception as e:
            print(f"Error processing sequence {seq}: {e}")
            continue
    
    return np.array(encodings)

def create_sequence_model(seq_length=33, vocab_size=21):
    """Create sequence model with attention"""
    # Input layer
    inputs = tf.keras.layers.Input(shape=(seq_length,))
    
    # Embedding layer
    x = tf.keras.layers.Embedding(vocab_size, 32)(inputs)
    
    # Parallel convolution blocks
    conv1 = tf.keras.layers.Conv1D(32, 3, padding='same')(x)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation('relu')(conv1)
    
    conv2 = tf.keras.layers.Conv1D(32, 5, padding='same')(x)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Activation('relu')(conv2)
    
    conv3 = tf.keras.layers.Conv1D(32, 7, padding='same')(x)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Activation('relu')(conv3)
    
    # Concatenate convolution results
    x = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
    
    # Bidirectional LSTM
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(x)
    
    # Self-attention mechanism
    attention = tf.keras.layers.Dense(1)(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention_weights = tf.keras.layers.Activation('softmax')(attention)
    attention_weights = tf.keras.layers.RepeatVector(64)(attention_weights)
    attention_weights = tf.keras.layers.Permute([2, 1])(attention_weights)
    
    # Apply attention
    x = tf.keras.layers.Multiply()([x, attention_weights])
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers with strong regularization
    x = tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Custom F1 Score metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

def train_model():
    try:
        # Load data
        print("Loading data...")
        train_df = pd.read_csv("./processed_data_train.csv")
        test_df = pd.read_csv("./processed_data_test.csv")
        
        # Prepare features
        print("\nPreparing sequence features...")
        X_train = prepare_sequence_data(train_df)
        X_test = prepare_sequence_data(test_df)
        
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        
        # Split training data to get a validation set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Calculate class weights
        total_samples = len(y_train)
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        
        pos_weight = (total_samples / (2 * n_pos))
        neg_weight = (total_samples / (2 * n_neg))
        class_weight = {0: neg_weight, 1: pos_weight}
        
        print(f"\nClass distribution in training set:")
        print(f"Positive samples: {n_pos} ({n_pos/total_samples*100:.2f}%)")
        print(f"Negative samples: {n_neg} ({n_neg/total_samples*100:.2f}%)")
        print(f"Class weights - Negative: {neg_weight:.2f}, Positive: {pos_weight:.2f}")
        
        # Create and compile model
        print("\nCreating model...")
        model = create_sequence_model()
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                F1Score(name='f1')
            ]
        )
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        
        # Callbacks
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',  # Changed to monitor AUC instead of F1
            patience=10,
            restore_best_weights=True,
            mode='max'
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_auc',  # Changed to monitor AUC instead of F1
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='max'
        )
        
        # Train model
        print("\nStarting training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        y_pred_proba = model.predict(X_test)
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
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion matrix:")
        print(cm)
        
        # Save predictions
        test_df['predictions'] = y_pred_proba
        test_df.to_csv('test_predictions_sequence.csv', index=False)
        
        return model, history
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    model, history = train_model()
    if model is not None:
        print("\nTraining completed successfully!")
    else:
        print("\nTraining failed!")