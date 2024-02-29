import numpy as np
import os
import mir_eval
from tqdm import tqdm


PROJECT_PATH = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

TRIPLE_METER_SONG = [
    34, 62, 102, 107, 152, 173, 176, 203, 215, 231, 254, 280, 307, 328, 369,
    584, 592, 653, 654, 662, 744, 749, 756, 770, 799, 843, 869, 872, 887
]

DATASET_PATH = os.path.join(PROJECT_PATH, 'data', 'pop909_w_structure_label')
LABEL_SOURCE = np.load(os.path.join(PROJECT_PATH, 'data', 'pop909_w_structure_label', 'label_source.npy'))


def _cum_time_to_time_dur(d):
    d_cumsum = np.cumsum(d)
    starts = np.insert(d_cumsum, 0, 0)[0: -1]
    return starts


def _parse_phrase_label(phrase_string):
    phrase_starts = [i for i, s in enumerate(phrase_string) if s.isalpha()] + [len(phrase_string)]
    phrase_names = [phrase_string[phrase_starts[i]: phrase_starts[i + 1]] for i in range(len(phrase_starts) - 1)]
    phrase_types = [pn[0] for pn in phrase_names]
    phrase_lgths = np.array([int(pn[1:]) for pn in phrase_names])
    return phrase_names, phrase_types, phrase_lgths


def read_label(label_fn):
    """label_fn is human_label1.txt or human_label1.txt in the song folder of the dataset."""
    with open(label_fn) as f:
        phrase_label = f.readlines()[0].strip()
    phrase_names, phrase_types, phrase_lgths = _parse_phrase_label(phrase_label)

    phrase_starts = _cum_time_to_time_dur(phrase_lgths)

    phrases = [{'name': pn, 'type': pt, 'lgth': pl, 'start': ps}
               for pn, pt, pl, ps in zip(phrase_names, phrase_types, phrase_lgths, phrase_starts)]

    return phrases


def read_melody(melody_fn):
    """melody_fn is melody.txt in the song folder of the dataset."""

    # convert txt file to numpy array of (pitch, duration)
    with open(melody_fn) as f:
        melody = f.readlines()
    melody = [m.strip().split(' ') for m in melody]
    melody = np.array([[int(m[0]), int(m[1])] for m in melody])

    # convert numpy array of (pitch, duration) to (onset, pitch, dur)
    starts = _cum_time_to_time_dur(melody[:, 1])
    durs = melody[:, 1]
    pitches = melody[:, 0]
    is_pitch = melody[:, 0] != 0
    note_mat = np.stack([starts, pitches, durs], -1)[is_pitch]

    return note_mat


def _read_chord_string(c):
    """
    "\x01"s are replaced with " ". ")"s are added to "sus4(b7".
    (E.g., check with if c_name[-7:] == 'sus4(b7'). And there may have other annotation problems.)
    The files in this repo have been cleaned.
    """
    c = c.strip().split(' ')
    c_name = c[0]
    c_dur = int(c[-1])

    # cleaning the chord symbol
    c_name = c_name.replace('\x01', '')

    # convert to chroma representation
    root, chroma, bass = mir_eval.chord.encode(c_name)
    chroma = np.roll(chroma, shift=root)
    return np.concatenate([np.array([root]), chroma, np.array([bass]), np.array([c_dur])])


def read_chord(chord_fn):
    """chord_fn is finalized_chord.txt in the song folder of the dataset."""
    with open(chord_fn) as f:
        chords = f.readlines()

    # convert chord text label to chroma representation
    chords = np.stack([_read_chord_string(c) for c in chords], 0)

    # convert chord to output chord matrix
    starts = _cum_time_to_time_dur(chords[:, -1])
    chord_mat = np.concatenate([starts[:, np.newaxis], chords], -1)

    return chord_mat


def read_data(data_fn, num_beat_per_measure=4, num_step_per_beat=4, song_name=None, label=1):
    if label == 1:
        label_fn = os.path.join(data_fn, 'human_label1.txt')
    elif label == 2:
        label_fn = os.path.join(data_fn, 'human_label2.txt')
    else:
        raise ValueError('Human label file does not exit.')

    label = read_label(label_fn)

    melody_fn = os.path.join(data_fn, 'melody.txt')
    melody = read_melody(melody_fn)  # convert txt file melody to note matrix

    chord_fn = os.path.join(data_fn, 'finalized_chord.txt')
    chord = read_chord(chord_fn)  # convert txt file chord to chord matrix

    return {'song_name': song_name,
            'melody': melody,
            'chord': chord,
            'phrase_label': label,
            'num_beat_per_measure': num_beat_per_measure,
            'num_step_per_beat': num_step_per_beat}


def read_pop909_dataset(song_ids=None):
    """Prepares pop909 dataset from song_ids selection. Returns a list of dicts."""

    dataset = []
    song_ids = [si for si in range(1, 910)] if song_ids is None else song_ids

    for i in tqdm(song_ids):
        label = LABEL_SOURCE[i - 1]  # which human label file to use

        num_beat_per_measure = 3 if i in TRIPLE_METER_SONG else 4

        song_name = str(i).zfill(3)  # e.g., '001'

        data_fn = os.path.join(DATASET_PATH, song_name)  # data folder of the song

        song_data = read_data(data_fn, num_beat_per_measure, 4, song_name, label)

        dataset.append(song_data)

    return dataset
