import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras import Input

BASE_STATE = 0
CAR_ONLY_STATE = 1
PEDESTRIAN_STATE = 2

def apply_optimal_action_rules(df):
    conditions = [
        (df['cars_L3'] == 0) & (df['peds_W1'] == 0) & (df['peds_W2'] == 0),                                 # Rule 1: Nothing on L3, W1, W2
        (df['cars_L1'] + df['cars_L2'] == 0) & (df['peds_W1'] + df['peds_W2'] == 0) & (df['cars_L3'] > 0),  # Rule 2: Only L3 cars
        (df['cars_L1'] + df['cars_L2'] == 0) & (df['peds_W1'] + df['peds_W2'] > 0),                         # Rule 3: No cars on L1/L2, but peds present on W1/W2
        (df['cars_L3'] > df['cars_L1'] + df['cars_L2']) & (df['peds_W1'] + df['peds_W2'] == 0),             # Rule 5: L3 > L1+L2 and no peds on W1/W2
        (df['wait_time_L3'] > df['wait_time_L1_L2']) & (df['peds_W1'] + df['peds_W2'] == 0),                # Rule 6: Wait time L3 > L1/L2, and no peds on W1/W2
        (df['cars_L3'] + df['peds_W1'] + df['peds_W2'] > df['cars_L1'] + df['cars_L2']),                    # Rule 4: Count L3 + W1/W2 > L1 + L2
        (df['wait_time_L3'] + df['wait_time_W1_W2'] > df['wait_time_L1_L2']),                               # Rule 7: Wait time L3 + W1/W2 > wait time L1+L2
    ]

    choices = [
        BASE_STATE,        # Rule 1
        CAR_ONLY_STATE,    # Rule 2
        PEDESTRIAN_STATE,  # Rule 3
        CAR_ONLY_STATE,    # Rule 5
        CAR_ONLY_STATE,    # Rule 6
        PEDESTRIAN_STATE,  # Rule 4
        PEDESTRIAN_STATE   # Rule 7
    ]

    return np.select(conditions, choices, default=BASE_STATE)

def generate_smart_traffic_data(n_samples):
    # Randomly generated data
    data = {
        'cars_L1': np.random.randint(0, 15, size=n_samples),
        'cars_L2': np.random.randint(0, 15, size=n_samples),
        'cars_L3': np.random.randint(0, 5, size=n_samples),
        'peds_W1': np.random.randint(0, 7, size=n_samples),
        'peds_W2': np.random.randint(0, 7, size=n_samples),
        'peds_W3': np.random.randint(0, 1, size=n_samples),
        'wait_time_L1_L2': np.random.randint(0, 30, size=n_samples),
        'wait_time_L3': np.random.randint(0, 30, size=n_samples),
        'wait_time_W1_W2': np.random.randint(0, 30, size=n_samples),
    }

    df = pd.DataFrame(data)

    # Define a list of hand-crafted data rows to append
    hardcoded_rows = [
        {
            'cars_L1': 0, 'cars_L2': 0, 'cars_L3': 3,
            'peds_W1': 0, 'peds_W2': 0, 'peds_W3': 0,
            'wait_time_L1_L2': 0, 'wait_time_L3': 10, 'wait_time_W1_W2': 0
        },  # Should go to CAR_ONLY_STATE via Rule 2 or 6

        {
            'cars_L1': 0, 'cars_L2': 0, 'cars_L3': 0,
            'peds_W1': 4, 'peds_W2': 5, 'peds_W3': 0,
            'wait_time_L1_L2': 0, 'wait_time_L3': 0, 'wait_time_W1_W2': 20
        },  # Should go to PEDESTRIAN_STATE via Rule 3 or 7

        {
            'cars_L1': 15, 'cars_L2': 10, 'cars_L3': 2,
            'peds_W1': 2, 'peds_W2': 0, 'peds_W3': 0,
            'wait_time_L1_L2': 20, 'wait_time_L3': 5, 'wait_time_W1_W2': 2
        },  # Should go to BASE_STATE due to heavy L1/L2 traffic

        {
            'cars_L1': 0, 'cars_L2': 0, 'cars_L3': 0,
            'peds_W1': 0, 'peds_W2': 0, 'peds_W3': 0,
            'wait_time_L1_L2': 0, 'wait_time_L3': 0, 'wait_time_W1_W2': 0
        },  # Should go to BASE_STATE via Rule 1

        {
            'cars_L1': 3, 'cars_L2': 4, 'cars_L3': 7,
            'peds_W1': 4, 'peds_W2': 5, 'peds_W3': 0,
            'wait_time_L1_L2': 20, 'wait_time_L3': 30, 'wait_time_W1_W2': 30
        },  # Should go to PEDESTRIAN_STATE via Rule 4 or 7
    ]

    hardcoded_df = pd.DataFrame(hardcoded_rows)
    full_df = pd.concat([df, hardcoded_df], ignore_index=True)

    full_df['optimal_state'] = apply_optimal_action_rules(full_df)

    return full_df

np.random.seed(42)
n_samples = 10000  # Increased samples for better training on complex rules
df = generate_smart_traffic_data(n_samples)

print((df['optimal_state'] == BASE_STATE).sum())
print((df['optimal_state'] == CAR_ONLY_STATE).sum())
print((df['optimal_state'] == PEDESTRIAN_STATE).sum())
# output in collab: 1565, 62, 8378

# Features are the sensor inputs the model will use to make decisions
features = [
    'cars_L1', 'cars_L2', 'cars_L3',
    'peds_W1', 'peds_W2', 'peds_W3',
    'wait_time_L1_L2', 'wait_time_L3', 'wait_time_W1_W2'
]
X = df[features]
# The label is the single state the model should predict
y = df['optimal_state']

# Normalizing data helps the model train more effectively
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("-" * 50)
print("COPY THESE VALUES INTO YOUR ANGULAR COMPONENT:")
print(f"Scaler Mean: {scaler.mean_.tolist()}")
print(f"Scaler Scale (Std Dev): {scaler.scale_.tolist()}")
print("-" * 50)

joblib.dump(scaler, 'smart_traffic_scaler.pkl')

# 'stratify=y' ensures that the distribution of states is the same in train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

input_layer = Input(shape=(X_train.shape[1],))
x = Dense(64, activation='relu')(input_layer)
x = Dense(64, activation='relu')(x)
output_layer = Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# Loss Function: 'sparse_categorical_crossentropy' is used for integer-based classification labels # Metrics: 'accuracy' is the key performance indicator for classification
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# train
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train,
          epochs=50,
          batch_size=32,
          validation_data=(X_val, y_val),
          callbacks=[early_stop])

# evaluate & save
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")
model.save('smart_traffic_classifier_model.h5')
"""
output of above:
COPY THESE VALUES INTO YOUR ANGULAR COMPONENT:
Scaler Mean: [6.994502748625687, 7.011694152923538, 1.9861069465267367, 3.0368815592203897, 3.049775112443778, 0.0, 14.593003498250875, 14.50064967516242, 14.522038980509745]
Scaler Scale (Std Dev): [4.325195499288317, 4.307050944032767, 1.4159464800406658, 1.985287985726915, 2.002776982486944, 1.0, 8.699760232927833, 8.675925814190926, 8.68162160910034]
--------------------------------------------------
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 9)]               0         
                                                                 
 dense (Dense)               (None, 64)                640       
                                                                 
 dense_1 (Dense)             (None, 64)                4160      
                                                                 
 dense_2 (Dense)             (None, 3)                 195       
                                                                 
=================================================================
Total params: 4995 (19.51 KB)
Trainable params: 4995 (19.51 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Epoch 1/50
251/251 [==============================] - 1s 3ms/step - loss: 0.2719 - accuracy: 0.9068 - val_loss: 0.1476 - val_accuracy: 0.9520
Epoch 2/50
251/251 [==============================] - 0s 2ms/step - loss: 0.1441 - accuracy: 0.9490 - val_loss: 0.1208 - val_accuracy: 0.9630
Epoch 3/50
251/251 [==============================] - 1s 2ms/step - loss: 0.1237 - accuracy: 0.9543 - val_loss: 0.1101 - val_accuracy: 0.9560
Epoch 4/50
251/251 [==============================] - 1s 3ms/step - loss: 0.1082 - accuracy: 0.9600 - val_loss: 0.0918 - val_accuracy: 0.9665
Epoch 5/50
251/251 [==============================] - 1s 4ms/step - loss: 0.0961 - accuracy: 0.9649 - val_loss: 0.0842 - val_accuracy: 0.9725
Epoch 6/50
251/251 [==============================] - 1s 4ms/step - loss: 0.0830 - accuracy: 0.9679 - val_loss: 0.0716 - val_accuracy: 0.9760
Epoch 7/50
251/251 [==============================] - 1s 5ms/step - loss: 0.0771 - accuracy: 0.9701 - val_loss: 0.0669 - val_accuracy: 0.9750
Epoch 8/50
251/251 [==============================] - 2s 7ms/step - loss: 0.0660 - accuracy: 0.9748 - val_loss: 0.0615 - val_accuracy: 0.9760
Epoch 9/50
251/251 [==============================] - 2s 6ms/step - loss: 0.0579 - accuracy: 0.9779 - val_loss: 0.0612 - val_accuracy: 0.9755
Epoch 10/50
251/251 [==============================] - 1s 4ms/step - loss: 0.0523 - accuracy: 0.9810 - val_loss: 0.0501 - val_accuracy: 0.9840
Epoch 11/50
251/251 [==============================] - 1s 4ms/step - loss: 0.0471 - accuracy: 0.9821 - val_loss: 0.0525 - val_accuracy: 0.9825
Epoch 12/50
251/251 [==============================] - 1s 4ms/step - loss: 0.0434 - accuracy: 0.9845 - val_loss: 0.0447 - val_accuracy: 0.9830
Epoch 13/50
251/251 [==============================] - 0s 2ms/step - loss: 0.0392 - accuracy: 0.9858 - val_loss: 0.0408 - val_accuracy: 0.9835
Epoch 14/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0366 - accuracy: 0.9886 - val_loss: 0.0421 - val_accuracy: 0.9845
Epoch 15/50
251/251 [==============================] - 0s 2ms/step - loss: 0.0351 - accuracy: 0.9881 - val_loss: 0.0382 - val_accuracy: 0.9860
Epoch 16/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0307 - accuracy: 0.9899 - val_loss: 0.0361 - val_accuracy: 0.9845
Epoch 17/50
251/251 [==============================] - 0s 2ms/step - loss: 0.0293 - accuracy: 0.9899 - val_loss: 0.0334 - val_accuracy: 0.9865
Epoch 18/50
251/251 [==============================] - 0s 2ms/step - loss: 0.0278 - accuracy: 0.9905 - val_loss: 0.0314 - val_accuracy: 0.9875
Epoch 19/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0235 - accuracy: 0.9929 - val_loss: 0.0307 - val_accuracy: 0.9860
Epoch 20/50
251/251 [==============================] - 0s 2ms/step - loss: 0.0235 - accuracy: 0.9921 - val_loss: 0.0391 - val_accuracy: 0.9860
Epoch 21/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0225 - accuracy: 0.9921 - val_loss: 0.0264 - val_accuracy: 0.9900
Epoch 22/50
251/251 [==============================] - 0s 2ms/step - loss: 0.0196 - accuracy: 0.9943 - val_loss: 0.0407 - val_accuracy: 0.9810
Epoch 23/50
251/251 [==============================] - 0s 2ms/step - loss: 0.0197 - accuracy: 0.9936 - val_loss: 0.0272 - val_accuracy: 0.9915
Epoch 24/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0169 - accuracy: 0.9953 - val_loss: 0.0286 - val_accuracy: 0.9880
Epoch 25/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0180 - accuracy: 0.9943 - val_loss: 0.0229 - val_accuracy: 0.9925
Epoch 26/50
251/251 [==============================] - 1s 3ms/step - loss: 0.0206 - accuracy: 0.9918 - val_loss: 0.0248 - val_accuracy: 0.9905
Epoch 27/50
251/251 [==============================] - 1s 3ms/step - loss: 0.0187 - accuracy: 0.9938 - val_loss: 0.0264 - val_accuracy: 0.9905
Epoch 28/50
251/251 [==============================] - 1s 3ms/step - loss: 0.0235 - accuracy: 0.9916 - val_loss: 0.0249 - val_accuracy: 0.9880
Epoch 29/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0122 - accuracy: 0.9969 - val_loss: 0.0224 - val_accuracy: 0.9905
Epoch 30/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0129 - accuracy: 0.9969 - val_loss: 0.0266 - val_accuracy: 0.9895
Epoch 31/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0131 - accuracy: 0.9965 - val_loss: 0.0219 - val_accuracy: 0.9900
Epoch 32/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0125 - accuracy: 0.9971 - val_loss: 0.0234 - val_accuracy: 0.9900
Epoch 33/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0112 - accuracy: 0.9974 - val_loss: 0.0259 - val_accuracy: 0.9895
Epoch 34/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0102 - accuracy: 0.9979 - val_loss: 0.0248 - val_accuracy: 0.9895
Epoch 35/50
251/251 [==============================] - 0s 2ms/step - loss: 0.0106 - accuracy: 0.9971 - val_loss: 0.0209 - val_accuracy: 0.9910
Epoch 36/50
251/251 [==============================] - 0s 2ms/step - loss: 0.0113 - accuracy: 0.9969 - val_loss: 0.0234 - val_accuracy: 0.9900
Epoch 37/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0128 - accuracy: 0.9951 - val_loss: 0.0217 - val_accuracy: 0.9915
Epoch 38/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0150 - accuracy: 0.9959 - val_loss: 0.0232 - val_accuracy: 0.9900
Epoch 39/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0089 - accuracy: 0.9983 - val_loss: 0.0177 - val_accuracy: 0.9945
Epoch 40/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0090 - accuracy: 0.9975 - val_loss: 0.0190 - val_accuracy: 0.9925
Epoch 41/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0074 - accuracy: 0.9985 - val_loss: 0.0166 - val_accuracy: 0.9935
Epoch 42/50
251/251 [==============================] - 0s 2ms/step - loss: 0.0072 - accuracy: 0.9991 - val_loss: 0.0257 - val_accuracy: 0.9885
Epoch 43/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0079 - accuracy: 0.9980 - val_loss: 0.0222 - val_accuracy: 0.9920
Epoch 44/50
251/251 [==============================] - 0s 2ms/step - loss: 0.0095 - accuracy: 0.9969 - val_loss: 0.0308 - val_accuracy: 0.9885
Epoch 45/50
251/251 [==============================] - 0s 2ms/step - loss: 0.0076 - accuracy: 0.9980 - val_loss: 0.0163 - val_accuracy: 0.9930
Epoch 46/50
251/251 [==============================] - 0s 2ms/step - loss: 0.0077 - accuracy: 0.9979 - val_loss: 0.0404 - val_accuracy: 0.9835
Epoch 47/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0074 - accuracy: 0.9981 - val_loss: 0.0187 - val_accuracy: 0.9925
Epoch 48/50
251/251 [==============================] - 1s 2ms/step - loss: 0.0108 - accuracy: 0.9970 - val_loss: 0.0372 - val_accuracy: 0.9865
Epoch 49/50
251/251 [==============================] - 1s 3ms/step - loss: 0.0060 - accuracy: 0.9985 - val_loss: 0.0160 - val_accuracy: 0.9945
Epoch 50/50
251/251 [==============================] - 1s 3ms/step - loss: 0.0050 - accuracy: 0.9991 - val_loss: 0.0262 - val_accuracy: 0.9900

Validation Accuracy: 99.00%
"""

# --- Example of Real-World Implementation ---

# 1. Get live sensor data (example)
current_sensor_data = {
    'cars_L1': [0], 'cars_L2': [0], 'cars_L3': [0],
    'peds_W1': [0], 'peds_W2': [0], 'peds_W3': [0],
    'wait_time_L1_L2': [0], 'wait_time_L3': [0], 'wait_time_W1_W2': [0]
}
live_df = pd.DataFrame(current_sensor_data)

# 2. Scale the data using the saved scaler
live_scaled = scaler.transform(live_df)

# 3. Predict the next state
predicted_probabilities = model.predict(live_scaled)
predicted_state = np.argmax(predicted_probabilities) # Get the state with the highest probability

# 4. Apply duration logic based on the predicted state
duration = 20 # Default duration

if predicted_state == CAR_ONLY_STATE:
    num_cars_l3 = live_df['cars_L3'].iloc[0]
    duration = 10 if num_cars_l3 > 3 else 5
    print(f"Entering CarOnlyState for {duration} seconds.")

elif predicted_state == PEDESTRIAN_STATE:
    num_peds = live_df['peds_W1'].iloc[0] + live_df['peds_W2'].iloc[0]
    duration = 25 if num_peds > 5 else 15
    print(f"Entering PedestrianState for {duration} seconds.")

else: # BaseState
    print("Entering BaseState.")

# The system would then set the lights for the predicted_state and run a timer for 'duration' seconds.

"""
output of above:
1/1 [==============================] - 0s 117ms/step
Entering BaseState.
"""

#below command to show model sumamry in collab
# from tensorflow.keras.models import load_model
# model = load_model("smart_traffic_classifier_model.h5")
# model.summary()
"""
output of above:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_3 (InputLayer)      │ (None, 9)              │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_9 (Dense)                 │ (None, 64)             │           640 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_10 (Dense)                │ (None, 64)             │         4,160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_11 (Dense)                │ (None, 3)              │           195 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 4,997 (19.52 KB)
 Trainable params: 4,995 (19.51 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 2 (12.00 B)
"""

# below is used to get the correct versions so the model can be converted to proper angular file to be used there
# !pip install tensorflowjs
# !pip install TensorFlow==2.15.0
# !pip install tensorflow-decision-forests==1.8.1

# then restart the session
# import tensorflow as tf
# tf.keras.backend.clear_session()

# then run this in the termainal to convert the model to tfjs format
# tensorflowjs_converter --input_format keras smart_traffic_classifier_model.h5 tfjs_model