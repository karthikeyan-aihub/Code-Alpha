import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from music21 import stream, note, tempo

# Suppress TensorFlow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load MIDI dataset or use your own dataset loading code
def load_dataset():
    # Load your dataset here
    # Example: Load MIDI files and convert them to the required format
    # For illustration, I'll generate random data
    X_train = np.random.rand(1000, 64, 128)  # Random data for illustration
    return X_train

# Define the RNN model
def build_rnn_model(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.TimeDistributed(keras.layers.Dense(128, activation='softmax'))
    ])
    return model

# Generate music sequences using the trained RNN model
def generate_music(model, length=100):
    # Seed sequence for generation
    seed = np.random.rand(1, length, 128)  # Assuming 128 features per timestep
    generated_sequence = model.predict(seed)
    return generated_sequence

# Convert the generated sequence to a MIDI file
def sequence_to_midi(sequence, output_file):
    s = stream.Stream()
    for timestep in sequence[0]:
        note_index = np.argmax(timestep)
        note_obj = note.Note(midi=note_index)  # Convert index to MIDI pitch
        s.append(note_obj)
    # Add tempo information
    s.insert(0, tempo.MetronomeMark(number=120))
    # Write the MIDI file
    s.write('midi', fp=output_file)

# Directory to save generated MIDI files
output_directory = r'C:\Users\karth\OneDrive\Documents\VS 1\Code Alpha\Task 3 - Music Generation with AI'

# Generate MIDI file
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load dataset
X_train = load_dataset()

# Build and compile the RNN model
model = build_rnn_model(input_shape=X_train.shape[1:])
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model (Replace this with your actual training code)
model.fit(X_train, X_train, batch_size=64, epochs=10)

# Generate music sequences
generated_sequence = generate_music(model, length=200)

# Save generated sequence to MIDI file
generated_midi_filename = os.path.join(output_directory, 'generated_rnn.mid')
sequence_to_midi(generated_sequence, generated_midi_filename)

print(f"Generated MIDI file: {generated_midi_filename}")

# Display the generated MIDI file in a GUI
generated_stream = stream.Stream()
for timestep in generated_sequence[0]:
    note_index = np.argmax(timestep)
    note_obj = note.Note(midi=note_index)
    generated_stream.append(note_obj)
generated_stream.show('midi')
