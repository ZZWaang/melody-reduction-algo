import os
from utils import read_pop909_dataset, output_to_midi, DEMO_FOLDER
from run_melody_reduction import run_melody_reduction_of_dataset


if __name__ == '__main__':
    song_ids = [1, 2, 3, 4, 5]  # song ids starting from 1
    dataset = read_pop909_dataset(song_ids=song_ids)

    os.makedirs(DEMO_FOLDER, exist_ok=True)
    num_path = 1
    bpm = 90
    plot_graph = False

    # dataset: a list of dictionary containing keys: 'song_name', 'melody', 'chord', 'phrase_label',
    # 'num_beat_per_measure' and 'num_step_per_beat'.
    run_melody_reduction_of_dataset(dataset, DEMO_FOLDER, num_path, plot_graph, bpm)
