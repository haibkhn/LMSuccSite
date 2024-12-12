import tensorflow as tf

print("TensorFlow version:", tf.__version__)
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm
from keras import Sequential

def prepare_data(df, window_size=1, shuffle_data=True):
    """Prepare structural features for model"""
    print(f"\nPreparing data from DataFrame with shape: {df.shape}")
    
    if shuffle_data:
        df = shuffle(df, random_state=42)
    
    # Process features with progress bar
    features_list = []
    print("\nProcessing features...")
    
    for _, row in df.iterrows():
        try:
            # Get angles (using list() to ensure proper conversion)
            phi = np.array(eval(str(row['phi_large'])))
            psi = np.array(eval(str(row['psi_large'])))
            
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
            print(f"Row content: {row}")
            continue
    
    X = np.array(features_list)
    y = np.array(df['label'].values)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y

# Main code
print("Starting script...")

try:
    # 1. Load and prepare data
    print("Loading data...")
    train_df = pd.read_csv("./processed_data_train.csv")
    test_df = pd.read_csv("./processed_data_test.csv")
    
    # Print first few rows to verify data
    print("\nFirst row of training data:")
    print(train_df.iloc[0])
    
    # 2. Prepare features
    X_train, y_train = prepare_data(train_df)
    X_test, y_test = prepare_data(test_df)
    
    # 3. Create simple model (like the test code)
    print("\nCreating model...")
    model = Sequential([
        # Embedding layer (useful if the input is sequential like text or tokens)
        tf.keras.layers.Embedding(input_dim=256, output_dim=21, input_length=X_train.shape[1]),
        
        # First convolutional layer (1D convolution for sequential data)
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.3),
        
        # Second convolutional layer
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.3),
        
        # Max pooling layer to reduce dimensionality
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Third convolutional layer
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Dropout(0.3),
        
        # Max pooling layer
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # Flatten the output of the last convolutional layer
        tf.keras.layers.Flatten(),
        
        # Fully connected layers after convolutions
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        
        # Output layer with a sigmoid activation for binary classification
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # 4. Compile model
    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # 5. Train model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # 6. Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test loss: {results[0]:.4f}")
    print(f"Test accuracy: {results[1]:.4f}")
    
    # 7. Confusion matrix
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print(cm)

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nScript completed.")