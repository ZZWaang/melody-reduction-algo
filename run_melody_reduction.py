import os
import numpy as np
from tqdm import tqdm
from tonal_reduction_algo.main import TrAlgo
from utils import McpMusic, output_to_midi


def run_melody_reduction_of_song(song_data, output_fn, num_path=1, plot_graph=False, bpm=90):
    """
    song_data: a dictionary containing keys: 'song_name', 'melody', 'chord', 'phrase_label',
        'num_beat_per_measure' and 'num_step_per_beat'.
    output_fn: output MIDI file name
    num_path: number of reduction samples (number of shortest paths to find)
    plot_graph: whether to show shortest-path visualization
    bpm: output MIDI bpm
    """
    # initialize algorithm
    tr_algo = TrAlgo()

    # the Melody-Chord-Phrase class stores input data dict more effectively
    mcp = McpMusic(melody=song_data['melody'],
                   chord=song_data['chord'],
                   phrase_label=song_data['phrase_label'],
                   num_beat_per_measure=song_data['num_beat_per_measure'],
                   num_step_per_beat=song_data['num_step_per_beat'],
                   song_name=song_data['song_name'],
                   clean_chord_unit=song_data['num_beat_per_measure'])
    song_phrases = mcp.segment_data_to_phrases()

    # initialize output lists
    song_note_mat = []  # the original melody
    song_chord_mat = []  # the original chord
    song_red_mats = [[] for _ in range(num_path)]  # the output reductions

    for phrase_id, p in enumerate(song_phrases):
        # run TRA
        phrase_note_mat, phrase_chord_mat, phrase_reduction_mats = \
            tr_algo.run(p['note_mat'], p['chord_mat'], p['start_measure'],
                        mcp.num_beat_per_measure, mcp.num_step_per_beat,
                        num_path, plot_graph)

        # prepend outputs
        song_note_mat.append(phrase_note_mat)
        song_chord_mat.append(phrase_chord_mat)
        for path_id in range(num_path):
            song_red_mats[path_id].append(phrase_reduction_mats[path_id])

    # save the result to a MIDI file
    whole_note_mat = np.concatenate(song_note_mat, 0)
    whole_chord_mat = np.concatenate(song_chord_mat, 0)
    whole_red_mats = [np.concatenate(rm, 0) for rm in song_red_mats]

    output_to_midi(output_fn, whole_note_mat, whole_chord_mat,
                   whole_red_mats, mcp.num_beat_per_measure, bpm=bpm)


def run_melody_reduction_of_dataset(dataset, output_folder, num_path=1, plot_graph=False, bpm=90):
    """
    dataset: a list of dictionary containing keys: 'song_name', 'melody', 'chord', 'phrase_label',
        'num_beat_per_measure' and 'num_step_per_beat'.
    output_folder: output folder to store generated MIDI files
    num_path: number of reduction samples (number of shortest paths to find)
    plot_graph: whether to show shortest-path visualization
    bpm: output MIDI bpm
    """
    for song_data in tqdm(dataset):
        output_fn = os.path.join(output_folder, song_data['song_name'] + '-red.mid')
        run_melody_reduction_of_song(song_data, output_fn, num_path, plot_graph, bpm=bpm)
