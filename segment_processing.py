from pydub import AudioSegment

import os
import wave
import glob
import spacy

PATH_SEP = '/'

nlp = spacy.load("en_core_web_sm")


class Output_ljspeech:

    def __init__(self, directory, metadata='metadata.csv', file_prefix='phrase_'):
        self.directory = directory
        self.metadata = f"{directory}/{metadata}"
        self.file_prefix = file_prefix
        self.indexes = [self.get_index(name[len(self.directory) + 6:]) for name in
                        glob.glob(f"{self.directory}{PATH_SEP}wavs{PATH_SEP}{self.file_prefix}*.wav")]
        if not os.path.exists(directory):
            os.makedirs(self.directory)
        if not os.path.exists(f"{self.directory}{PATH_SEP}wavs"):
            os.makedirs(f"{self.directory}{PATH_SEP}wavs")

    def get_index(self, name):
        return int(name[len(self.file_prefix):-4])

    def next_index(self):
        self.indexes = [self.get_index(name[len(self.directory) + 6:]) for name in
                        glob.glob(f"{self.directory}{PATH_SEP}wavs{PATH_SEP}{self.file_prefix}*.wav")]
        if len(self.indexes) > 0:
            return max(self.indexes) + 1
        else:
            return 1

    def file_name(self, num):
        return f"{self.file_prefix}{num}"

    def file_path(self, num):
        return self.directory + '{PATH_SEP}wavs' + '{PATH_SEP}' + self.file_name(num) + '.wav'

    def add_record(self, record, text):

        index = self.next_index()
        file_name = self.file_name(index)
        record.export(self.file_path(index), format="wav")
        with open(self.metadata, 'a') as file:
            file.write(file_name + '|' + text + '|' + text.lower() + '\n')

        return index


def get_filesegment(start_position, num_ms, input_filepath, output_filepath='temp.wav', output_frame_rate=22050):
    wav_file = wave.open(input_filepath, 'rb')

    frame_rate = wav_file.getframerate()

    chunk_size = frame_rate * num_ms // 1000
    print(f"Chunk size: {chunk_size}")

    start_pos = start_position * frame_rate // 1000

    print(f"Setting start position {start_pos}, reading {chunk_size} frames")
    wav_file.setpos(start_pos)
    data = wav_file.readframes(chunk_size)
    print(len(data))
    wav_file.close()

    output_file = wave.open(output_filepath, 'wb')

    params = wav_file.getparams()
    output_file.setparams(params)
    output_file.writeframes(data)

    output_file.close()

    input_file = AudioSegment.from_file(output_filepath)
    new_frame_rate = 22050
    resampled_audio = input_file.set_frame_rate(output_frame_rate)
    resampled_audio.export(output_filepath, format='wav')

    return output_filepath


def process_segment(file, output, model):
    global file_num
    result = model.transcribe(file, word_timestamps=True)
    audio_record = AudioSegment.from_wav(file)

    phrases, offset = get_phrases(result)

    if offset < 0:
        wav_file = wave.open(file, 'rb')
        num_frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        wav_file.close()
        length = num_frames * 1000 // frame_rate
        print(f"Nothing to add from the segment, length of the file {length}")
        return length

    num_phrases = len(phrases)
    if num_phrases < 2:
        print(f"Nothing to add from the segment, returning offset {offset * 1000}")
        return offset * 1000

    print(f"Total phrases {num_phrases}")

    for i, segment in enumerate(phrases):
        if len(segment['text'].split()) < 2:
            continue
        start = segment['start']
        end = segment['end']
        audio_segment = audio_record[start * 1000: min(end * 1000 + 70, len(audio_record) - 1)]
        file_index = output.add_record(audio_segment, segment['text'])

        print(f"\n Phrase No {file_index}: {segment['text']}\nStart: {start} End: {end}")
        # display(Audio(output.file_path(file_index)))
    return offset * 1000


def get_word_list(transcription):
    words = []
    for segment in transcription['segments']:
        for word in segment['words']:
            words.append({'text': word['word'], 'start': word['start'], 'end': word['end'], })
    return words


def get_phrases(transcription):
    sents = nlp(transcription['text']).sents

    phrases = []
    words = get_word_list(transcription)
    num_words = len(words)
    if num_words == 0:
        return phrases, -1
    words = iter(words)

    for sent in sents:
        sent_len = len(str(sent).split())

        if sent_len == 0:
            continue

        word = next(words)
        start = word['start']
        end = word['end']

        for i in range(sent_len - 1):
            try:
                word = next(words)
                end = word['end']
            except Exception as er:
                print(er)
                print(f"Sentence that caused an error: {sent}")
                print(f"Transcription: {transcription['text']}, total words: {num_words}")
                return phrases, end

        if word['text'].strip() != str(sent).split()[-1].strip():
            print(f"!!!! Phrase is not aligned !!!! {word['text']} vs {str(sent).split()[-1]}")
            # phrases.pop()
            return phrases, end

        phrases.append({'text': str(sent), 'start': start, 'end': end})
        # print(f"Sentence: \n{sent},\nLast word : {word}")
    phrases.pop()
    return phrases, start