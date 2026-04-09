"""Tests for the midi_adapter compatibility layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from miditok.midi_adapter import (
    AdapterControlChange,
    AdapterKeySignature,
    AdapterNote,
    AdapterPedal,
    AdapterPitchBend,
    AdapterScore,
    AdapterTempo,
    AdapterTimeSignature,
    AdapterTrack,
    ControlChangeList,
    KeySignatureList,
    NoteList,
    PedalList,
    PitchBendList,
    TempoList,
    TimeSignatureList,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_MIDI = REPO_ROOT / "violin_partita_bwv-1004_1_(c)grossman.mid"
SAMPLE_MIDI_2 = REPO_ROOT / "tests" / "MIDIs_one_track" / "Maestro_1.mid"


@pytest.fixture
def score() -> AdapterScore:
    return AdapterScore.from_file(SAMPLE_MIDI)


def test_load_score_from_file(score: AdapterScore) -> None:
    assert score.ticks_per_quarter > 0
    assert len(score.tracks) > 0
    assert score.tracks[0].notes


def test_score_constructed_from_path_str() -> None:
    s = AdapterScore(str(SAMPLE_MIDI))
    assert s.note_num() > 0


def test_score_constructed_from_pathlib() -> None:
    s = AdapterScore(SAMPLE_MIDI)
    assert s.note_num() > 0


def test_empty_score() -> None:
    s = AdapterScore(ticks_per_quarter=480)
    assert s.ticks_per_quarter == 480
    assert s.tracks == []
    assert s.note_num() == 0
    assert s.end() == 0


def test_tempo_extraction(score: AdapterScore) -> None:
    assert len(score.tempos) > 0
    for t in score.tempos:
        assert isinstance(t, AdapterTempo)
        assert t.tempo > 0


def test_time_signature_extraction(score: AdapterScore) -> None:
    assert len(score.time_signatures) > 0
    ts = score.time_signatures[0]
    assert ts.numerator > 0
    assert ts.denominator > 0


def test_round_trip_dump(tmp_path: Path, score: AdapterScore) -> None:
    out = tmp_path / "round.mid"
    score.dump_midi(out)
    reloaded = AdapterScore.from_file(out)
    assert reloaded.note_num() == score.note_num()


def test_note_list_numpy_shapes() -> None:
    notes = NoteList()
    for i in range(5):
        notes.append(AdapterNote(i * 100, 50, 60 + i, 90))
    arr = notes.numpy()
    assert arr["time"].dtype == np.int32
    assert arr["duration"].dtype == np.int32
    assert arr["pitch"].dtype == np.int32
    assert arr["velocity"].dtype == np.int32
    assert arr["time"].shape == (5,)


def test_note_list_from_numpy_round_trip() -> None:
    notes = NoteList()
    for i in range(7):
        notes.append(AdapterNote(i, i + 1, 60, 80))
    arr = notes.numpy()
    rebuilt = NoteList.from_numpy(**arr)
    assert list(notes) == list(rebuilt)


def test_note_on_off_pairing() -> None:
    track = AdapterTrack(name="t")
    track.notes.append(AdapterNote(0, 100, 60, 90))
    track.notes.append(AdapterNote(100, 50, 64, 80))
    assert len(track.notes) == 2
    nat = track.to_native()
    # NoteOn + NoteOff per note
    on = sum(1 for ev in nat.events if ev.event_type == "NoteOn")
    off = sum(1 for ev in nat.events if ev.event_type == "NoteOff")
    assert on == 2 and off == 2


def test_orphan_note_on_default_duration() -> None:
    import midi_toolkit as mt

    nt = mt.Track("orphan")
    nt.add_event(mt.Event.note_on(0, 0, 60, 100))
    nt.add_event(mt.Event.note_off(50, 0, 64, 0))  # different pitch — won't match
    nt.add_event(mt.Event.note_on(100, 0, 64, 80))
    nt.add_event(mt.Event.note_off(150, 0, 64, 0))
    track = AdapterTrack(_native=nt)
    assert any(n.duration == 1 for n in track.notes)


def test_score_resample_changes_tick_values(score: AdapterScore) -> None:
    new_tpq = score.ticks_per_quarter * 2
    new = score.resample(new_tpq)
    assert new.ticks_per_quarter == new_tpq
    if score.tracks[0].notes and new.tracks[0].notes:
        original_first = score.tracks[0].notes[0]
        new_first = new.tracks[0].notes[0]
        assert new_first.time == original_first.time * 2


def test_score_resample_min_dur() -> None:
    s = AdapterScore(ticks_per_quarter=480)
    t = AdapterTrack()
    t.notes.append(AdapterNote(0, 4, 60, 80))
    s.tracks = [t]
    new = s.resample(60, min_dur=5)
    assert new.tracks[0].notes[0].duration >= 5


def test_score_copy_independence(score: AdapterScore) -> None:
    other = score.copy()
    other.tracks[0].notes.append(AdapterNote(99999, 1, 60, 80))
    assert len(other.tracks[0].notes) != len(score.tracks[0].notes)


def test_score_end_returns_max_tick(score: AdapterScore) -> None:
    end = score.end()
    assert end > 0
    assert end >= max(n.end for n in score.tracks[0].notes)


def test_tempo_bpm_mspq_conversion() -> None:
    t = AdapterTempo(0, 120.0)
    assert abs(t.mspq - 500_000) < 5


def test_track_program_extraction() -> None:
    s = AdapterScore.from_file(SAMPLE_MIDI)
    assert all(isinstance(t.program, int) for t in s.tracks)


def test_track_is_drum_detection() -> None:
    import midi_toolkit as mt

    nt = mt.Track("drums")
    nt.add_event(mt.Event("ProgramChange", 0, program=0, channel=9))
    nt.add_event(mt.Event.note_on(0, 9, 36, 100))
    nt.add_event(mt.Event.note_off(50, 9, 36, 0))
    s = AdapterScore(ticks_per_quarter=480)
    s._native.tracks = [nt]
    s._tracks = s._build_tracks_from_native()
    assert s._tracks[0].is_drum is True


def test_typed_lists_numpy_round_trip() -> None:
    tl = TempoList()
    tl.append(AdapterTempo(0, 120.0))
    tl.append(AdapterTempo(480, 90.0))
    arr = tl.numpy()
    assert arr["time"].dtype == np.int32
    rebuilt = TempoList.from_numpy(**arr)
    assert list(tl) == list(rebuilt)

    tsl = TimeSignatureList([AdapterTimeSignature(0, 4, 4)])
    arr = tsl.numpy()
    assert TimeSignatureList.from_numpy(**arr) == tsl

    pl = PedalList([AdapterPedal(0, 100)])
    assert PedalList.from_numpy(**pl.numpy()) == pl

    pbl = PitchBendList([AdapterPitchBend(0, 1024)])
    assert PitchBendList.from_numpy(**pbl.numpy()) == pbl

    cl = ControlChangeList([AdapterControlChange(0, 64, 100)])
    assert ControlChangeList.from_numpy(**cl.numpy()) == cl

    ksl = KeySignatureList([AdapterKeySignature(0, 0, 0)])
    assert KeySignatureList.from_numpy(**ksl.numpy()) == ksl


def test_clip_score(score: AdapterScore) -> None:
    end = score.end()
    clipped = score.clip(0, end // 2)
    assert clipped.note_num() <= score.note_num()
    assert clipped.end() <= end


def test_score_eq() -> None:
    s1 = AdapterScore.from_file(SAMPLE_MIDI)
    s2 = AdapterScore.from_file(SAMPLE_MIDI)
    assert s1 == s2


def test_dump_midi_preserves_tempos(tmp_path: Path, score: AdapterScore) -> None:
    out = tmp_path / "tempos.mid"
    score.dump_midi(out)
    reloaded = AdapterScore.from_file(out)
    assert len(reloaded.tempos) > 0
