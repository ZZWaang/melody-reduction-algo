import numpy as np


class McpMusic:

    """MCP for Melody-Chord-Phrase. A class to store a piece containing melody, chord, and phrase labels."""

    def __init__(self, melody, chord, phrase_label, num_beat_per_measure=4,
                 num_step_per_beat=4, song_name=None,
                 clean_chord_unit=4):
        self.song_name = song_name

        # structural attributes
        self.num_beat_per_measure = num_beat_per_measure
        self.num_step_per_beat = num_step_per_beat

        self.phrase_names = \
            np.array([pl['name'] for pl in phrase_label])
        self.phrase_types = \
            np.array([pl['type'] for pl in phrase_label])
        self.phrase_starts = \
            np.array([pl['start'] for pl in phrase_label])
        self.phrase_lengths = \
            np.array([pl['lgth'] for pl in phrase_label])

        self.num_phrases = len(phrase_label)
        self.num_measures = sum(self.phrase_lengths)
        self.num_beats = self.num_measures * num_beat_per_measure
        self.num_steps = self.num_beats * num_step_per_beat

        # melody and chord data
        self.melody = melody
        self.chord = chord

        # ensuring chord having a maximum duration of self.clearn_chord_unit
        self.clean_chord_unit = clean_chord_unit
        self._clean_chord()

        # determine piece length from phrase label, melody, and chord input
        self.total_measure = self._compute_total_measure()
        self.total_beat = self.total_measure * self.num_beat_per_measure

        # pad chord (= last chord) to match self.total_beat
        self._regularize_chord()

        # pad phrase with label 'z' to match self.total_measure
        self._regularize_phrases()

        # compute song_dict for phrase-wise segmentation
        self.song_dict = None
        self._create_song_level_dict()

    def _compute_total_measure(self):
        # propose candidates from phrase, chord and melody
        last_step = (self.melody[:, 0] + self.melody[:, 2]).max()
        num_measure0 = \
            int(np.ceil(last_step / self.num_step_per_beat /
                        self.num_beat_per_measure))

        last_beat = (self.chord[:, 0] + self.chord[:, -1]).max()
        num_measure1 = int(np.ceil(last_beat / self.num_beat_per_measure))

        num_measure2 = sum(self.phrase_lengths)
        return max(num_measure0, num_measure1, num_measure2)

    def _regularize_chord(self):
        chord = self.chord
        end_time = (self.chord[:, 0] + self.chord[:, -1]).max()
        fill_n_beat = self.total_beat - end_time
        if fill_n_beat == 0:
            return

        pad_durs = [self.clean_chord_unit] * (fill_n_beat // self.clean_chord_unit)
        if fill_n_beat - sum(pad_durs) > 0:
            pad_durs = [fill_n_beat] + pad_durs
        for d in pad_durs:
            stack_chord = chord[-1].copy()
            stack_chord[0] = chord[-1, 0] + chord[-1, -1]
            stack_chord[-1] = d

            chord = np.concatenate([chord, stack_chord[np.newaxis, :]], 0)
        self.chord = chord

    def _regularize_phrases(self):
        original_phrase_length = sum(self.phrase_lengths)
        if self.total_measure == original_phrase_length:
            return

        extra_phrase_length = self.total_measure - original_phrase_length
        extra_phrase_name = 'z' + str(extra_phrase_length)

        self.phrase_names = np.append(self.phrase_names, extra_phrase_name)
        self.phrase_types = np.append(self.phrase_types, 'z')
        self.phrase_lengths = np.append(self.phrase_lengths,
                                        extra_phrase_length)
        self.phrase_starts = np.append(self.phrase_starts,
                                       original_phrase_length)

    def _clean_chord(self):
        new_chords = []
        n_chord = len(self.chord)

        for i in range(n_chord):
            chord_start = self.chord[i, 0]
            chord_dur = self.chord[i, -1]

            cum_dur = 0
            s = chord_start

            while cum_dur < chord_dur:
                d = min(self.clean_chord_unit - s % self.clean_chord_unit, chord_dur - cum_dur)
                c = self.chord[i].copy()
                c[0] = s
                c[-1] = d
                new_chords.append(c)
                s = s + d
                cum_dur += d
        new_chords = np.stack(new_chords, 0)
        self.chord = new_chords

    def phrase_to_beat(self, phrase):
        start_measure = self.phrase_starts[phrase]
        end_measure = self.phrase_lengths[phrase] + start_measure
        start_beat = start_measure * self.num_beat_per_measure
        end_beat = end_measure * self.num_beat_per_measure
        return start_beat, end_beat

    def create_a_phrase_level_dict(self, phrase_id):
        start_measure = self.phrase_starts[phrase_id]
        phrase_length = self.phrase_lengths[phrase_id]
        end_measure = start_measure + phrase_length
        phrase_dict = {
            'phrase_name': self.phrase_names[phrase_id],
            'phrase_type': self.phrase_types[phrase_id],
            'phrase_length': self.phrase_lengths[phrase_id],
            'start_measure': start_measure,
            'end_measure': end_measure,
            'length': phrase_length,
            'mel_slice': None,
            'chd_slice': None,
        }
        return phrase_dict

    def _create_song_level_dict(self):
        self.song_dict = {
            'song_name': self.song_name,
            'total_phrase': self.num_phrases,
            'total_measure': self.num_measures,
            'total_beat': self.num_beats,
            'total_step': self.num_steps,
            'phrases': [self.create_a_phrase_level_dict(phrase_id)
                        for phrase_id in range(self.num_phrases)]
        }
        self._fill_phrase_level_slices()

    def _fill_phrase_level_mel_slices(self):
        n_note = self.melody.shape[0]

        onset_beats = self.melody[:, 0] // self.num_step_per_beat

        current_ind = 0
        for phrase_id, phrase in enumerate(self.song_dict['phrases']):
            start_beat, end_beat = self.phrase_to_beat(phrase_id)
            for i in range(current_ind, n_note):
                if onset_beats[i] >= end_beat:
                    phrase[f'mel_slice'] = slice(current_ind, i)
                    current_ind = i
                    break
            else:
                phrase[f'mel_slice'] = slice(current_ind, n_note)
                current_ind = n_note

    def _fill_phrase_level_chd_slices(self):
        n_chord = self.chord.shape[0]
        current_ind = 0
        for phrase_id, phrase in enumerate(self.song_dict['phrases']):
            start_beat, end_beat = self.phrase_to_beat(phrase_id)
            for i in range(current_ind, n_chord):
                if self.chord[i, 0] >= end_beat:
                    phrase['chd_slice'] = slice(current_ind, i)
                    current_ind = i
                    break
            else:
                phrase['chd_slice'] = slice(current_ind, n_chord)
                current_ind = n_chord

    def _fill_phrase_level_slices(self):
        self._fill_phrase_level_mel_slices()
        self._fill_phrase_level_chd_slices()

    def segment_data_to_phrases(self):
        data = []
        for phrase in self.song_dict['phrases']:
            mel_slice = phrase['mel_slice']
            chd_slice = phrase['chd_slice']

            melody_phrase = self.melody[mel_slice]
            chord_phrase = self.chord[chd_slice].copy()

            n_note = melody_phrase.shape[0]

            start_measure = phrase['start_measure']

            n_measure = phrase['length']

            data.append({'note_mat': melody_phrase, 'chord_mat': chord_phrase,
                         'n_measure': n_measure, 'start_measure': start_measure})

        return data
