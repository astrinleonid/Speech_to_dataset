import whisper

file = 'Audio_data/ted_example.waw'
model = whisper.load_model("base")
result = model.transcribe(file, word_timestamps=True)