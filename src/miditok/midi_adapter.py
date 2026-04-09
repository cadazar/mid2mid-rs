"""Adapter layer wrapping ``midi_toolkit`` (Rust/PyO3) with a symusic-compatible API.

This module provides drop-in replacements for the ``symusic`` types that the
MidiTok tokenizers consume. The goal is to expose just enough surface area for
the tokenizers to encode and decode MIDI files end-to-end without requiring
``symusic`` itself.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

import midi_toolkit as _mt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUSTAIN_CC = 64
_PITCH_BEND_TYPE = "PitchBend"

# ---------------------------------------------------------------------------
# Note + typed list classes
# ---------------------------------------------------------------------------


def _build_adapter_classmethods() -> None:
    """Attach symusic-style ``from_numpy`` classmethods to scalar types.

    The bindings are inserted at the bottom of the module after the typed
    list classes are declared.
    """


class AdapterNote:
    """A note represented by ``time``, ``duration``, ``pitch``, ``velocity``."""

    __slots__ = ("time", "duration", "pitch", "velocity")

    def __init__(self, time: int, duration: int, pitch: int, velocity: int) -> None:
        self.time = int(time)
        self.duration = int(duration)
        self.pitch = int(pitch)
        self.velocity = int(velocity)

    @property
    def start(self) -> int:
        return self.time

    @property
    def end(self) -> int:
        return self.time + self.duration

    def copy(self) -> "AdapterNote":
        return AdapterNote(self.time, self.duration, self.pitch, self.velocity)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdapterNote):
            return NotImplemented
        return (
            self.time == other.time
            and self.duration == other.duration
            and self.pitch == other.pitch
            and self.velocity == other.velocity
        )

    def __hash__(self) -> int:
        return hash((self.time, self.duration, self.pitch, self.velocity))

    def __repr__(self) -> str:
        return (
            f"AdapterNote(time={self.time}, duration={self.duration}, "
            f"pitch={self.pitch}, velocity={self.velocity})"
        )


class _TypedList(list):
    """Base class for typed lists with sort/copy/numpy helpers."""

    item_cls: type = object

    def sort(self, key: Callable[[Any], Any] | None = None, reverse: bool = False) -> None:  # type: ignore[override]
        if key is None:
            key = lambda x: getattr(x, "time", 0)
        super().sort(key=key, reverse=reverse)

    def copy(self):  # type: ignore[override]
        cls = type(self)
        return cls(item.copy() if hasattr(item, "copy") else copy.copy(item) for item in self)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, list):
            return NotImplemented
        if len(self) != len(other):
            return False
        return all(a == b for a, b in zip(self, other))

    def __ne__(self, other: object) -> bool:  # type: ignore[override]
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return eq
        return not eq

    __hash__ = None  # type: ignore[assignment]


class NoteList(_TypedList):
    item_cls = AdapterNote

    def numpy(self) -> dict[str, np.ndarray]:
        n = len(self)
        time = np.empty(n, dtype=np.int32)
        duration = np.empty(n, dtype=np.int32)
        pitch = np.empty(n, dtype=np.int32)
        velocity = np.empty(n, dtype=np.int32)
        for i, note in enumerate(self):
            time[i] = note.time
            duration[i] = note.duration
            pitch[i] = note.pitch
            velocity[i] = note.velocity
        return {
            "time": time,
            "duration": duration,
            "pitch": pitch,
            "velocity": velocity,
        }

    @classmethod
    def from_numpy(
        cls,
        time: np.ndarray,
        duration: np.ndarray,
        pitch: np.ndarray,
        velocity: np.ndarray,
    ) -> "NoteList":
        out = cls()
        for t, d, p, v in zip(time, duration, pitch, velocity):
            out.append(AdapterNote(int(t), int(d), int(p), int(v)))
        return out


class AdapterTempo:
    __slots__ = ("time", "tempo")

    def __init__(self, time: int, tempo: float) -> None:
        self.time = int(time)
        self.tempo = float(tempo)

    @property
    def qpm(self) -> float:  # alias used by some symusic code
        return self.tempo

    @property
    def mspq(self) -> int:
        return int(round(60_000_000 / self.tempo)) if self.tempo > 0 else 500_000

    def copy(self) -> "AdapterTempo":
        return AdapterTempo(self.time, self.tempo)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdapterTempo):
            return NotImplemented
        return self.time == other.time and abs(self.tempo - other.tempo) < 1e-6

    def __hash__(self) -> int:
        return hash((self.time, round(self.tempo, 4)))

    def __repr__(self) -> str:
        return f"AdapterTempo(time={self.time}, tempo={self.tempo})"


class TempoList(_TypedList):
    item_cls = AdapterTempo

    def numpy(self) -> dict[str, np.ndarray]:
        n = len(self)
        time = np.empty(n, dtype=np.int32)
        qpm = np.empty(n, dtype=np.float64)
        for i, t in enumerate(self):
            time[i] = t.time
            qpm[i] = t.tempo
        return {"time": time, "qpm": qpm}

    @classmethod
    def from_numpy(cls, time: np.ndarray, qpm: np.ndarray) -> "TempoList":
        out = cls()
        for t, q in zip(time, qpm):
            out.append(AdapterTempo(int(t), float(q)))
        return out


class AdapterTimeSignature:
    __slots__ = ("time", "numerator", "denominator")

    def __init__(self, time: int, numerator: int, denominator: int) -> None:
        self.time = int(time)
        self.numerator = int(numerator)
        self.denominator = int(denominator)

    def copy(self) -> "AdapterTimeSignature":
        return AdapterTimeSignature(self.time, self.numerator, self.denominator)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdapterTimeSignature):
            return NotImplemented
        return (
            self.time == other.time
            and self.numerator == other.numerator
            and self.denominator == other.denominator
        )

    def __hash__(self) -> int:
        return hash((self.time, self.numerator, self.denominator))

    def __repr__(self) -> str:
        return (
            f"AdapterTimeSignature(time={self.time}, numerator={self.numerator}, "
            f"denominator={self.denominator})"
        )


class TimeSignatureList(_TypedList):
    item_cls = AdapterTimeSignature

    def numpy(self) -> dict[str, np.ndarray]:
        n = len(self)
        time = np.empty(n, dtype=np.int32)
        numerator = np.empty(n, dtype=np.int32)
        denominator = np.empty(n, dtype=np.int32)
        for i, ts in enumerate(self):
            time[i] = ts.time
            numerator[i] = ts.numerator
            denominator[i] = ts.denominator
        return {"time": time, "numerator": numerator, "denominator": denominator}

    @classmethod
    def from_numpy(
        cls,
        time: np.ndarray,
        numerator: np.ndarray,
        denominator: np.ndarray,
    ) -> "TimeSignatureList":
        out = cls()
        for t, n, d in zip(time, numerator, denominator):
            out.append(AdapterTimeSignature(int(t), int(n), int(d)))
        return out


class AdapterKeySignature:
    __slots__ = ("time", "key", "tonality")

    def __init__(self, time: int, key: int, tonality: int) -> None:
        self.time = int(time)
        self.key = int(key)
        self.tonality = int(tonality)

    def copy(self) -> "AdapterKeySignature":
        return AdapterKeySignature(self.time, self.key, self.tonality)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdapterKeySignature):
            return NotImplemented
        return (
            self.time == other.time
            and self.key == other.key
            and self.tonality == other.tonality
        )

    def __hash__(self) -> int:
        return hash((self.time, self.key, self.tonality))


class KeySignatureList(_TypedList):
    item_cls = AdapterKeySignature

    def numpy(self) -> dict[str, np.ndarray]:
        n = len(self)
        time = np.empty(n, dtype=np.int32)
        key = np.empty(n, dtype=np.int32)
        tonality = np.empty(n, dtype=np.int32)
        for i, k in enumerate(self):
            time[i] = k.time
            key[i] = k.key
            tonality[i] = k.tonality
        return {"time": time, "key": key, "tonality": tonality}

    @classmethod
    def from_numpy(
        cls,
        time: np.ndarray,
        key: np.ndarray,
        tonality: np.ndarray,
    ) -> "KeySignatureList":
        out = cls()
        for t, k, s in zip(time, key, tonality):
            out.append(AdapterKeySignature(int(t), int(k), int(s)))
        return out


class AdapterPedal:
    __slots__ = ("time", "duration")

    def __init__(self, time: int, duration: int) -> None:
        self.time = int(time)
        self.duration = int(duration)

    @property
    def end(self) -> int:
        return self.time + self.duration

    def copy(self) -> "AdapterPedal":
        return AdapterPedal(self.time, self.duration)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdapterPedal):
            return NotImplemented
        return self.time == other.time and self.duration == other.duration

    def __hash__(self) -> int:
        return hash((self.time, self.duration))


class PedalList(_TypedList):
    item_cls = AdapterPedal

    def numpy(self) -> dict[str, np.ndarray]:
        n = len(self)
        time = np.empty(n, dtype=np.int32)
        duration = np.empty(n, dtype=np.int32)
        for i, p in enumerate(self):
            time[i] = p.time
            duration[i] = p.duration
        return {"time": time, "duration": duration}

    @classmethod
    def from_numpy(cls, time: np.ndarray, duration: np.ndarray) -> "PedalList":
        out = cls()
        for t, d in zip(time, duration):
            out.append(AdapterPedal(int(t), int(d)))
        return out


class AdapterPitchBend:
    __slots__ = ("time", "value")

    def __init__(self, time: int, value: int) -> None:
        self.time = int(time)
        self.value = int(value)

    def copy(self) -> "AdapterPitchBend":
        return AdapterPitchBend(self.time, self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdapterPitchBend):
            return NotImplemented
        return self.time == other.time and self.value == other.value

    def __hash__(self) -> int:
        return hash((self.time, self.value))


class PitchBendList(_TypedList):
    item_cls = AdapterPitchBend

    def numpy(self) -> dict[str, np.ndarray]:
        n = len(self)
        time = np.empty(n, dtype=np.int32)
        value = np.empty(n, dtype=np.int32)
        for i, pb in enumerate(self):
            time[i] = pb.time
            value[i] = pb.value
        return {"time": time, "value": value}

    @classmethod
    def from_numpy(cls, time: np.ndarray, value: np.ndarray) -> "PitchBendList":
        out = cls()
        for t, v in zip(time, value):
            out.append(AdapterPitchBend(int(t), int(v)))
        return out


class AdapterControlChange:
    __slots__ = ("time", "number", "value")

    def __init__(self, time: int, number: int, value: int) -> None:
        self.time = int(time)
        self.number = int(number)
        self.value = int(value)

    def copy(self) -> "AdapterControlChange":
        return AdapterControlChange(self.time, self.number, self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdapterControlChange):
            return NotImplemented
        return (
            self.time == other.time
            and self.number == other.number
            and self.value == other.value
        )

    def __hash__(self) -> int:
        return hash((self.time, self.number, self.value))


class ControlChangeList(_TypedList):
    item_cls = AdapterControlChange

    def numpy(self) -> dict[str, np.ndarray]:
        n = len(self)
        time = np.empty(n, dtype=np.int32)
        number = np.empty(n, dtype=np.int32)
        value = np.empty(n, dtype=np.int32)
        for i, c in enumerate(self):
            time[i] = c.time
            number[i] = c.number
            value[i] = c.value
        return {"time": time, "number": number, "value": value}

    @classmethod
    def from_numpy(
        cls,
        time: np.ndarray,
        number: np.ndarray,
        value: np.ndarray,
    ) -> "ControlChangeList":
        out = cls()
        for t, n, v in zip(time, number, value):
            out.append(AdapterControlChange(int(t), int(n), int(v)))
        return out


# ---------------------------------------------------------------------------
# Track adapter
# ---------------------------------------------------------------------------


def _pair_notes_from_events(events: Iterable[Any]) -> NoteList:
    """Pair NoteOn / NoteOff events into AdapterNotes.

    Each NoteOn is matched with the next NoteOff sharing the same pitch and
    channel.  Orphaned NoteOns get a default duration of 1 tick.
    """
    notes = NoteList()
    pending: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for ev in events:
        et = ev.event_type
        if et == "NoteOn" and ev.velocity > 0:
            key = (ev.channel, ev.pitch)
            pending.setdefault(key, []).append((ev.tick, ev.velocity))
        elif et == "NoteOff" or (et == "NoteOn" and ev.velocity == 0):
            key = (ev.channel, ev.pitch)
            stack = pending.get(key)
            if stack:
                start, vel = stack.pop(0)
                notes.append(
                    AdapterNote(start, max(1, ev.tick - start), ev.pitch, vel)
                )
    # orphans
    for (ch, pitch), stack in pending.items():
        for start, vel in stack:
            notes.append(AdapterNote(start, 1, pitch, vel))
    notes.sort()
    return notes


def _is_drum_channel(ch: int) -> bool:
    # MIDI channel 10 (1-indexed) = channel 9 (0-indexed)
    return ch == 9


class AdapterTrack:
    """A symusic-compatible Track holding notes/CCs/etc. as Python lists.

    The track may also retain a reference to a midi_toolkit Track that
    contributed the events; once we extract notes/CCs we treat the
    Python-side lists as the source of truth.
    """

    def __init__(
        self,
        program: int = 0,
        is_drum: bool = False,
        name: str = "",
        _native: Any | None = None,
    ) -> None:
        self._program = int(program)
        self._is_drum = bool(is_drum)
        self._name = str(name)
        self.notes: NoteList = NoteList()
        self.controls: ControlChangeList = ControlChangeList()
        self.pitch_bends: PitchBendList = PitchBendList()
        self.pedals: PedalList = PedalList()
        self.lyrics: list = []
        self.markers: list = []
        if _native is not None:
            self._populate_from_native(_native)

    def _populate_from_native(self, native: Any) -> None:
        if not self._name:
            self._name = native.name or ""
        self.notes = _pair_notes_from_events(native.events)
        for ev in native.events:
            et = ev.event_type
            if et == "ControlChange":
                if ev.controller == _SUSTAIN_CC:
                    continue  # handled below as pedals
                self.controls.append(AdapterControlChange(ev.tick, ev.controller, ev.value))
            elif et == _PITCH_BEND_TYPE:
                self.pitch_bends.append(AdapterPitchBend(ev.tick, ev.value))
        # Pedals from sustain CC events
        starts: list[int] = []
        for ev in native.events:
            if ev.event_type == "ControlChange" and ev.controller == _SUSTAIN_CC:
                if ev.value >= 64:
                    starts.append(ev.tick)
                elif starts:
                    s = starts.pop(0)
                    self.pedals.append(AdapterPedal(s, max(1, ev.tick - s)))
        for s in starts:
            self.pedals.append(AdapterPedal(s, 1))
        self.controls.sort()
        self.pitch_bends.sort()
        self.pedals.sort()

    # -- properties --------------------------------------------------------
    @property
    def program(self) -> int:
        return self._program

    @program.setter
    def program(self, value: int) -> None:
        self._program = int(value)

    @property
    def is_drum(self) -> bool:
        return self._is_drum

    @is_drum.setter
    def is_drum(self, value: bool) -> None:
        self._is_drum = bool(value)

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = str(value)

    # -- helpers -----------------------------------------------------------
    def copy(self) -> "AdapterTrack":
        new = AdapterTrack(program=self._program, is_drum=self._is_drum, name=self._name)
        new.notes = self.notes.copy()
        new.controls = self.controls.copy()
        new.pitch_bends = self.pitch_bends.copy()
        new.pedals = self.pedals.copy()
        new.lyrics = list(self.lyrics)
        new.markers = list(self.markers)
        return new

    __copy__ = copy
    __deepcopy__ = lambda self, memo: self.copy()

    def end(self) -> int:
        max_t = 0
        if self.notes:
            max_t = max(n.end for n in self.notes)
        if self.controls:
            max_t = max(max_t, max(c.time for c in self.controls))
        if self.pitch_bends:
            max_t = max(max_t, max(pb.time for pb in self.pitch_bends))
        if self.pedals:
            max_t = max(max_t, max(p.end for p in self.pedals))
        return max_t

    def note_num(self) -> int:
        return len(self.notes)

    def __len__(self) -> int:
        return self.note_num()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdapterTrack):
            return NotImplemented
        return (
            self._program == other._program
            and self._is_drum == other._is_drum
            and self._name == other._name
            and list(self.notes) == list(other.notes)
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"AdapterTrack(name={self._name!r}, program={self._program}, "
            f"is_drum={self._is_drum}, notes={self.note_num()})"
        )

    def to_native(self) -> Any:
        """Build a fresh midi_toolkit Track from the current state."""
        nt = _mt.Track(self._name or "")
        ch = 9 if self._is_drum else 0
        nt.add_event(_mt.Event("ProgramChange", 0, program=int(self._program), channel=ch))
        for c in self.controls:
            nt.add_event(_mt.Event("ControlChange", int(c.time), channel=ch,
                                   controller=int(c.number), value=int(c.value)))
        for pb in self.pitch_bends:
            nt.add_event(_mt.Event(_PITCH_BEND_TYPE, int(pb.time), channel=ch, value=int(pb.value)))
        for p in self.pedals:
            nt.add_event(_mt.Event("ControlChange", int(p.time), channel=ch,
                                   controller=_SUSTAIN_CC, value=127))
            nt.add_event(_mt.Event("ControlChange", int(p.time + p.duration), channel=ch,
                                   controller=_SUSTAIN_CC, value=0))
        for note in self.notes:
            nt.add_event(_mt.Event.note_on(int(note.time), ch, int(note.pitch), int(note.velocity)))
            nt.add_event(_mt.Event.note_off(int(note.time + note.duration), ch,
                                            int(note.pitch), int(note.velocity)))
        nt.sort_events()
        return nt


def _clone_event(ev: Any) -> Any:
    et = ev.event_type
    if et == "NoteOn":
        return _mt.Event.note_on(ev.tick, ev.channel, ev.pitch, ev.velocity)
    if et == "NoteOff":
        return _mt.Event.note_off(ev.tick, ev.channel, ev.pitch, ev.velocity)
    kwargs: dict[str, Any] = {}
    for attr in ("channel", "pitch", "velocity", "controller", "value", "program",
                 "tempo", "numerator", "denominator", "key", "scale"):
        try:
            v = getattr(ev, attr)
            if v is not None:
                kwargs[attr] = v
        except Exception:
            pass
    # denominator is interpreted (e.g. 4 not 2). Need to convert back to power-of-2 exponent
    # for the constructor? The constructor accepts the interpreted value too based on tests.
    return _mt.Event(et, ev.tick, **kwargs)


# ---------------------------------------------------------------------------
# Score adapter
# ---------------------------------------------------------------------------


class AdapterScore:
    """A symusic-compatible Score wrapping a midi_toolkit Score."""

    def __init__(self, arg: Any = None, ticks_per_quarter: int | None = None) -> None:
        # symusic-style overloads:
        #   AdapterScore() -> empty
        #   AdapterScore(480) -> empty with tpq
        #   AdapterScore("path.mid") -> load
        #   AdapterScore(Path("path.mid")) -> load
        if isinstance(arg, (str, Path)):
            self._native = _mt.Score.from_file(str(arg))
        elif isinstance(arg, _mt.Score):
            self._native = arg
        elif isinstance(arg, int):
            self._native = _mt.Score(1, arg)
        elif arg is None:
            tpq = ticks_per_quarter if ticks_per_quarter is not None else 480
            self._native = _mt.Score(1, tpq)
        else:
            raise TypeError(f"Cannot construct AdapterScore from {type(arg)!r}")

        # cache adapter tracks (extract programs from native tracks)
        self._tracks: list[AdapterTrack] = self._build_tracks_from_native()
        self._tempos: TempoList = self._extract_tempos()
        self._time_signatures: TimeSignatureList = self._extract_time_signatures()
        self._key_signatures: KeySignatureList = self._extract_key_signatures()
        self._markers: list = []

    # -- factory ----------------------------------------------------------
    @staticmethod
    def from_file(path: str | Path) -> "AdapterScore":
        return AdapterScore(str(path))

    # -- track extraction -------------------------------------------------
    def _build_tracks_from_native(self) -> list[AdapterTrack]:
        out: list[AdapterTrack] = []
        for tr in self._native.tracks:
            program = 0
            channel = 0
            for ev in tr.events:
                if ev.event_type == "ProgramChange":
                    program = ev.program
                    channel = ev.channel
                    break
                if ev.event_type in ("NoteOn", "NoteOff", "ControlChange"):
                    channel = ev.channel
            is_drum = _is_drum_channel(channel)
            out.append(AdapterTrack(
                program=program,
                is_drum=is_drum,
                name=tr.name or "",
                _native=tr,
            ))
        return out

    def clip(self, start: int, end: int, clip_end: bool = True) -> "AdapterScore":  # noqa: ARG002
        """Return a copy of this score restricted to ``[start, end)``."""
        new = AdapterScore(ticks_per_quarter=self.ticks_per_quarter)
        new._tracks = []
        for tr in self._tracks:
            nt = AdapterTrack(program=tr.program, is_drum=tr.is_drum, name=tr.name)
            for note in tr.notes:
                if note.time >= start and note.time < end:
                    nt.notes.append(AdapterNote(note.time - start, note.duration, note.pitch, note.velocity))
            for c in tr.controls:
                if start <= c.time < end:
                    nt.controls.append(AdapterControlChange(c.time - start, c.number, c.value))
            for pb in tr.pitch_bends:
                if start <= pb.time < end:
                    nt.pitch_bends.append(AdapterPitchBend(pb.time - start, pb.value))
            for p in tr.pedals:
                if start <= p.time < end:
                    nt.pedals.append(AdapterPedal(p.time - start, p.duration))
            new._tracks.append(nt)
        new._tempos = TempoList(
            AdapterTempo(t.time - start, t.tempo)
            for t in self._tempos if start <= t.time < end
        )
        new._time_signatures = TimeSignatureList(
            AdapterTimeSignature(t.time - start, t.numerator, t.denominator)
            for t in self._time_signatures if start <= t.time < end
        )
        new._key_signatures = KeySignatureList(
            AdapterKeySignature(k.time - start, k.key, k.tonality)
            for k in self._key_signatures if start <= k.time < end
        )
        return new

    def _extract_tempos(self) -> TempoList:
        out = TempoList()
        seen: set[tuple[int, int]] = set()
        for tr in self._native.tracks:
            for ev in tr.events:
                if ev.event_type == "Tempo":
                    key = (ev.tick, ev.tempo)
                    if key in seen:
                        continue
                    seen.add(key)
                    bpm = 60_000_000 / ev.tempo if ev.tempo > 0 else 120.0
                    out.append(AdapterTempo(ev.tick, bpm))
        out.sort()
        return out

    def _extract_time_signatures(self) -> TimeSignatureList:
        out = TimeSignatureList()
        seen: set[tuple[int, int, int]] = set()
        for tr in self._native.tracks:
            for ev in tr.events:
                if ev.event_type == "TimeSignature":
                    key = (ev.tick, ev.numerator, ev.denominator)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(AdapterTimeSignature(ev.tick, ev.numerator, ev.denominator))
        out.sort()
        return out

    def _extract_key_signatures(self) -> KeySignatureList:
        out = KeySignatureList()
        seen: set[tuple[int, int, int]] = set()
        for tr in self._native.tracks:
            for ev in tr.events:
                if ev.event_type == "KeySignature":
                    key = (ev.tick, ev.key, ev.scale)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(AdapterKeySignature(ev.tick, ev.key, ev.scale))
        out.sort()
        return out

    # -- properties -------------------------------------------------------
    @property
    def ticks_per_quarter(self) -> int:
        return self._native.ticks_per_quarter

    @ticks_per_quarter.setter
    def ticks_per_quarter(self, value: int) -> None:
        self._native.ticks_per_quarter = int(value)

    # symusic uses ``tpq`` and ``ticks_per_quarter`` interchangeably
    @property
    def tpq(self) -> int:
        return self.ticks_per_quarter

    @property
    def tracks(self) -> list[AdapterTrack]:
        return self._tracks

    @tracks.setter
    def tracks(self, tracks: Iterable[AdapterTrack]) -> None:
        new_tracks = list(tracks)
        self._tracks = new_tracks
        self._sync_native_tracks()

    def _sync_native_tracks(self) -> None:
        self._native.tracks = [t.to_native() for t in self._tracks]

    @property
    def tempos(self) -> TempoList:
        return self._tempos

    @tempos.setter
    def tempos(self, tempos: Iterable[AdapterTempo]) -> None:
        self._tempos = TempoList(tempos)

    @property
    def time_signatures(self) -> TimeSignatureList:
        return self._time_signatures

    @time_signatures.setter
    def time_signatures(self, tss: Iterable[AdapterTimeSignature]) -> None:
        self._time_signatures = TimeSignatureList(tss)

    @property
    def key_signatures(self) -> KeySignatureList:
        return self._key_signatures

    @key_signatures.setter
    def key_signatures(self, kss: Iterable[AdapterKeySignature]) -> None:
        self._key_signatures = KeySignatureList(kss)

    @property
    def markers(self) -> list:
        return self._markers

    @markers.setter
    def markers(self, markers: Iterable) -> None:
        self._markers = list(markers)

    # -- helpers ----------------------------------------------------------
    def end(self) -> int:
        max_tick = 0
        for t in self._tracks:
            max_tick = max(max_tick, t.end())
        for tempos_attr in (self._tempos, self._time_signatures, self._key_signatures):
            for item in tempos_attr:
                max_tick = max(max_tick, item.time)
        return max_tick

    def note_num(self) -> int:
        return sum(t.note_num() for t in self._tracks)

    def copy(self) -> "AdapterScore":
        new = AdapterScore.__new__(AdapterScore)
        new._native = _mt.Score(self._native.format_type, self._native.ticks_per_quarter)
        new._tracks = [t.copy() for t in self._tracks]
        new._tempos = TempoList(t.copy() for t in self._tempos)
        new._time_signatures = TimeSignatureList(t.copy() for t in self._time_signatures)
        new._key_signatures = KeySignatureList(k.copy() for k in self._key_signatures)
        new._markers = list(self._markers)
        return new

    __copy__ = copy
    __deepcopy__ = lambda self, memo: self.copy()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdapterScore):
            return NotImplemented
        return (
            self.ticks_per_quarter == other.ticks_per_quarter
            and list(self._tracks) == list(other._tracks)
            and list(self._tempos) == list(other._tempos)
            and list(self._time_signatures) == list(other._time_signatures)
        )

    def shift_time(self, offset: int) -> "AdapterScore":
        """Return a copy with all event times shifted by ``offset``.

        Negative offsets move events earlier; times are clamped to >= 0.
        """
        new = self.copy()
        for tr in new._tracks:
            for n in tr.notes:
                n.time = max(0, n.time + offset)
            for c in tr.controls:
                c.time = max(0, c.time + offset)
            for pb in tr.pitch_bends:
                pb.time = max(0, pb.time + offset)
            for p in tr.pedals:
                p.time = max(0, p.time + offset)
        new._tempos = TempoList(
            AdapterTempo(max(0, t.time + offset), t.tempo) for t in new._tempos
        )
        new._time_signatures = TimeSignatureList(
            AdapterTimeSignature(max(0, t.time + offset), t.numerator, t.denominator)
            for t in new._time_signatures
        )
        new._key_signatures = KeySignatureList(
            AdapterKeySignature(max(0, k.time + offset), k.key, k.tonality)
            for k in new._key_signatures
        )
        return new

    # -- resampling -------------------------------------------------------
    def resample(self, new_tpq: int, min_dur: int = 1) -> "AdapterScore":
        old_tpq = self.ticks_per_quarter
        ratio = new_tpq / old_tpq
        new = AdapterScore(ticks_per_quarter=new_tpq)
        new_tracks: list[AdapterTrack] = []
        for tr in self._tracks:
            adapted = AdapterTrack(program=tr.program, is_drum=tr.is_drum, name=tr.name)
            for n in tr.notes:
                new_time = int(round(n.time * ratio))
                new_dur = max(min_dur, int(round(n.duration * ratio)))
                adapted.notes.append(AdapterNote(new_time, new_dur, n.pitch, n.velocity))
            for c in tr.controls:
                adapted.controls.append(
                    AdapterControlChange(int(round(c.time * ratio)), c.number, c.value)
                )
            for pb in tr.pitch_bends:
                adapted.pitch_bends.append(
                    AdapterPitchBend(int(round(pb.time * ratio)), pb.value)
                )
            for p in tr.pedals:
                adapted.pedals.append(
                    AdapterPedal(int(round(p.time * ratio)), max(min_dur, int(round(p.duration * ratio))))
                )
            new_tracks.append(adapted)
        new._tracks = new_tracks
        new._sync_native_tracks()
        new._tempos = TempoList(
            AdapterTempo(int(round(t.time * ratio)), t.tempo) for t in self._tempos
        )
        new._time_signatures = TimeSignatureList(
            AdapterTimeSignature(int(round(t.time * ratio)), t.numerator, t.denominator)
            for t in self._time_signatures
        )
        new._key_signatures = KeySignatureList(
            AdapterKeySignature(int(round(k.time * ratio)), k.key, k.tonality)
            for k in self._key_signatures
        )
        new._markers = list(self._markers)
        return new

    # -- export -----------------------------------------------------------
    def dump_midi(self, path: str | Path) -> None:
        out = _mt.Score(self._native.format_type or 1, self.ticks_per_quarter)
        meta_events: list[Any] = []
        for tempo in self._tempos:
            mspq = int(round(60_000_000 / tempo.tempo)) if tempo.tempo > 0 else 500_000
            meta_events.append(_mt.Event("Tempo", int(tempo.time), tempo=mspq))
        for ts in self._time_signatures:
            meta_events.append(
                _mt.Event(
                    "TimeSignature", int(ts.time),
                    numerator=int(ts.numerator), denominator=int(ts.denominator),
                )
            )
        for ks in self._key_signatures:
            meta_events.append(
                _mt.Event(
                    "KeySignature", int(ks.time),
                    key=int(ks.key), scale=int(ks.tonality),
                )
            )
        meta_events.sort(key=lambda e: e.tick)

        new_tracks: list[Any] = []
        for i, tr in enumerate(self._tracks):
            nt = tr.to_native()
            if i == 0:
                # Insert meta events at the start
                existing = list(nt.events)
                nt = _mt.Track(tr.name or "")
                for ev in meta_events:
                    nt.add_event(_clone_event(ev))
                for ev in existing:
                    nt.add_event(ev)
                nt.sort_events()
            new_tracks.append(nt)
        if not new_tracks and meta_events:
            nt = _mt.Track("")
            for ev in meta_events:
                nt.add_event(_clone_event(ev))
            new_tracks.append(nt)
        out.tracks = new_tracks
        out.to_file(str(path))

    def to_file(self, path: str | Path) -> None:
        self.dump_midi(path)

    def __repr__(self) -> str:
        return (
            f"AdapterScore(tpq={self.ticks_per_quarter}, tracks={len(self._tracks)}, "
            f"notes={self.note_num()})"
        )


# Attach class-level ``from_numpy`` helpers so that callers can write
# ``Note.from_numpy(time=..., ...)`` returning a typed list (mirrors symusic).
AdapterNote.from_numpy = classmethod(  # type: ignore[attr-defined]
    lambda cls, time, duration, pitch, velocity: NoteList.from_numpy(
        time, duration, pitch, velocity
    )
)
AdapterTempo.from_numpy = classmethod(  # type: ignore[attr-defined]
    lambda cls, time, qpm: TempoList.from_numpy(time, qpm)
)
AdapterTimeSignature.from_numpy = classmethod(  # type: ignore[attr-defined]
    lambda cls, time, numerator, denominator: TimeSignatureList.from_numpy(
        time, numerator, denominator
    )
)
AdapterKeySignature.from_numpy = classmethod(  # type: ignore[attr-defined]
    lambda cls, time, key, tonality: KeySignatureList.from_numpy(time, key, tonality)
)
AdapterPedal.from_numpy = classmethod(  # type: ignore[attr-defined]
    lambda cls, time, duration: PedalList.from_numpy(time, duration)
)
AdapterPitchBend.from_numpy = classmethod(  # type: ignore[attr-defined]
    lambda cls, time, value: PitchBendList.from_numpy(time, value)
)
AdapterControlChange.from_numpy = classmethod(  # type: ignore[attr-defined]
    lambda cls, time, number, value: ControlChangeList.from_numpy(time, number, value)
)


# Convenience aliases matching symusic naming
Note = AdapterNote
Score = AdapterScore
Track = AdapterTrack
Tempo = AdapterTempo
TimeSignature = AdapterTimeSignature
KeySignature = AdapterKeySignature
Pedal = AdapterPedal
PitchBend = AdapterPitchBend
ControlChange = AdapterControlChange
TextMeta = None  # not used by core tokenizers; provided for import compatibility
