import pretty_midi as pm


DEMO_FOLDER = './tra_demo'


def note_mat_to_notes(note_mat, bpm=120.):
    alpha = 60 / bpm * 0.25

    return [pm.Note(80, int(p), s * alpha, (s + d) * alpha)
            for s, p, d in note_mat]


def chord_mat_to_notes(chord_mat, bpm=120.):
    alpha = 60 / bpm
    notes = []

    for chord in chord_mat:
        start = chord[0] * alpha
        end = (chord[0] + chord[-1]) * alpha
        bass = (chord[1] + chord[14]) % 12 + 36

        pitches = [bass]
        for i in range(12):
            if chord[2 + i] == 1:
                pitches.append(i + 48)
        for p in pitches:
            notes.append(pm.Note(80, int(p), start, end))
    return notes


def output_to_midi(output_fn, note_mat, chord_mat, reduction_mats, nbpm,
                   bpm=90.):
    midi = pm.PrettyMIDI(initial_tempo=bpm)

    midi.time_signature_changes.append(pm.TimeSignature(nbpm, 4, 0.))

    ins0 = pm.Instrument(0, name='melody')
    ins0.notes = note_mat_to_notes(note_mat[:, 0: 3], bpm)
    midi.instruments.append(ins0)

    ins1 = pm.Instrument(0, name='chord')
    ins1.notes = chord_mat_to_notes(chord_mat, bpm)
    midi.instruments.append(ins1)

    for i, red_mat in enumerate(reduction_mats):
        ins = pm.Instrument(0, name=f'reduction-{i}')
        ins.notes = note_mat_to_notes(red_mat[:, 0: 3], bpm)
        midi.instruments.append(ins)

    midi.write(output_fn)
