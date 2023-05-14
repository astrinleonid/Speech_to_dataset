from segment_processing import Output_ljspeech, get_filesegment, process_segment, PATH_SEP

import whisper
import os
import argparse
import glob
import wave
# import librosa
# import surfboard



model = whisper.load_model("base")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", type=str,
                    help="Provide the name of the directory")
parser.add_argument("-f", "--dataset_folder", type=str,
                    help="Provide the name of the dataset folder")

def process_audiofile(filepath, output, model):
    # Get the number of frames and the frame rate of the WAV file
    wav_file = wave.open(filepath, 'rb')
    num_frames = wav_file.getnframes()
    frame_rate = wav_file.getframerate()
    wav_file.close()

    index_path = filepath[:-3] + 'index'
    if not os.path.exists(index_path):
        current_pos = 0
        with open(index_path, "w") as file:
            file.write(str(current_pos))

    else:
        with open(index_path) as file:
            current_pos = int(file.read())

    length_in_ms = num_frames * 1000 // frame_rate

    while length_in_ms > current_pos + 1000:
        seg_len = min(180000, length_in_ms - current_pos)
        print(
            f"File length, s, {length_in_ms / 1000} Current position, s, {current_pos / 1000}, Getting next segment with length, ms {seg_len}")
        file = get_filesegment(current_pos, seg_len, filepath)
        increment = int(process_segment(file, output, model))
        current_pos += increment
        if increment == 0:
            print(f"Stuck on position {current_pos}")
            break
        print(f"Total length {length_in_ms} Current position {current_pos}, Increment: {increment}")
        with open(index_path, "w") as file:
            file.write(str(current_pos))

if __name__ == "__main__":

    args = parser.parse_args()
    directory = args.directory
    dataset_folder = args.dataset_folder
    print(directory, dataset_folder)
    metadata_file = "metadata.csv"
    temp_file = "temp.wav"

    files = glob.glob(directory + f"{PATH_SEP}*.wav")
    for file in files:
        prefix = file[:-4]
        output = Output_ljspeech(dataset_folder, metadata_file, prefix)
        process_audiofile(file, output, model)