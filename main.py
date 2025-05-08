
import os
import numpy as np
from pretty_midi import PrettyMIDI, Instrument, Note
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from IPython.display import Audio, display
import soundfile as sf

# ---------- Setup ----------
MAX_FILES = 5  # Only process 5 MIDI files
SEQUENCE_LENGTH = 30

# ---------- Download Sample Data ----------
print("Downloading sample data...")
!wget -q https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
!unzip -q maestro-v3.0.0-midi.zip -d /content/midi_files

# ---------- Load MIDI Files (Only 5 files) ----------
print(f"\nLoading first {MAX_FILES} MIDI files...")
midi_files = []
count = 0
for root, _, files in os.walk("/content/midi_files/maestro-v3.0.0"):
    for file in files:
        if count >= MAX_FILES:
            break
        if file.endswith((".midi", ".mid")):
            try:
                midi = PrettyMIDI(os.path.join(root, file))
                midi_files.append(midi)
                count += 1
                print(f"Loaded: {file}")
            except:
                continue

# ---------- Prepare Training Data ----------
print("\nPreparing training data...")
notes = []
for midi in midi_files:
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append(note.pitch)

unique_notes = sorted(set(notes))
note_to_int = {note: i for i, note in enumerate(unique_notes)}
n_vocab = len(unique_notes)

X = []
y = []
for i in range(len(notes) - SEQUENCE_LENGTH):
    seq_in = notes[i:i + SEQUENCE_LENGTH]
    seq_out = notes[i + SEQUENCE_LENGTH]
    X.append([note_to_int[note] for note in seq_in])
    y.append(note_to_int[seq_out])

X = np.reshape(X, (len(X), SEQUENCE_LENGTH, 1))
X = X / float(n_vocab)
y = to_categorical(y)

# ---------- Build Model ----------
print("\nBuilding model...")
model = Sequential([
    LSTM(128, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(n_vocab, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ---------- Train Model ----------
print("\nTraining model...")
history = model.fit(X, y, epochs=30, batch_size=64, validation_split=0.2)

# ---------- Generate Music ----------
print("\nGenerating new music...")
def generate_music(start_seq, length=100):
    generated = []
    current_seq = start_seq.copy()
    for _ in range(length):
        prediction = model.predict(current_seq.reshape(1, SEQUENCE_LENGTH, 1))
        predicted_note = np.argmax(prediction)
        generated.append(predicted_note)
        current_seq = np.append(current_seq[1:], predicted_note/n_vocab)
    return generated

start_idx = np.random.randint(0, len(X)-1)
start_seq = X[start_idx] * n_vocab
generated_notes = generate_music(start_seq, length=150)

# ---------- Save and Play ----------
print("\nSaving and playing result...")
pm = PrettyMIDI()
instrument = Instrument(program=0)

start_time = 0.0
for note_idx in generated_notes:
    pitch = unique_notes[note_idx]
    note = Note(
        velocity=np.random.randint(80, 110),
        pitch=pitch,
        start=start_time,
        end=start_time + 0.5
    )
    instrument.notes.append(note)
    start_time += 0.5

pm.instruments.append(instrument)
pm.write("/content/generated.mid")

# Convert to WAV and play
!fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 /content/generated.mid -F /content/output.wav -r 44100 > /dev/null 2>&1
audio_data, sample_rate = sf.read('/content/output.wav')
print("✅ Music generated successfully! Playing...")
display(Audio(audio_data, rate=sample_rate))