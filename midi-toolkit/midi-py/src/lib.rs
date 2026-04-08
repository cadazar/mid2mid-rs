use pyo3::prelude::*;
use midi_core::{self, EventData};

#[pyclass]
#[derive(Clone)]
pub struct Event {
    pub inner: midi_core::Event,
}

#[pymethods]
impl Event {
    #[new]
    #[pyo3(signature = (event_type, absolute_tick, channel=0, pitch=0, velocity=0, controller=0, value=0, program=0, tempo=500000, numerator=4, denominator=4, clocks_per_click=24, n_32nd_notes=8, key=0, scale=0, text=""))]
    fn new(
        event_type: &str,
        absolute_tick: u32,
        channel: u8, pitch: u8, velocity: u8,
        controller: u8, value: u16, program: u8,
        tempo: u32,
        numerator: u8, denominator: u32, clocks_per_click: u8, n_32nd_notes: u8,
        key: i8, scale: u8,
        text: &str,
    ) -> PyResult<Self> {
        let den = (denominator as f32).log2() as u8;
        let data = match event_type {
            "NoteOn" => EventData::NoteOn { ch: channel, pitch, vel: velocity },
            "NoteOff" => EventData::NoteOff { ch: channel, pitch, vel: velocity },
            "ControlChange" => EventData::ControlChange { ch: channel, cc: controller, val: value as u8 },
            "ProgramChange" => EventData::ProgramChange { ch: channel, prog: program },
            "PitchBend" => EventData::PitchBend { ch: channel, val: value },
            "Tempo" => EventData::Tempo { usec_per_quarter: tempo },
            "TimeSignature" => EventData::TimeSignature { num: numerator, den, clocks_per_click, n_32nd_notes },
            "KeySignature" => EventData::KeySignature { sf: key, mi: scale },
            "TrackName" => EventData::TrackName { text: text.as_bytes().to_vec() },
            "EndOfTrack" => EventData::EndOfTrack,
            _ => EventData::OtherMeta { type_: 0, data: vec![] },
        };
        Ok(Event { inner: midi_core::Event { delta_tick: 0, absolute_tick, data, has_status_byte: true } })
    }

    #[staticmethod]
    fn note_on(tick: u32, channel: u8, pitch: u8, velocity: u8) -> Self {
        let data = EventData::NoteOn { ch: channel, pitch, vel: velocity };
        Event { inner: midi_core::Event { delta_tick: 0, absolute_tick: tick, data, has_status_byte: true } }
    }

    #[staticmethod]
    fn note_off(tick: u32, channel: u8, pitch: u8, velocity: u8) -> Self {
        let data = EventData::NoteOff { ch: channel, pitch, vel: velocity };
        Event { inner: midi_core::Event { delta_tick: 0, absolute_tick: tick, data, has_status_byte: true } }
    }

    #[getter]
    fn event_type(&self) -> String {
        match &self.inner.data {
            EventData::NoteOn { vel, .. } => if *vel == 0 { "NoteOff".to_string() } else { "NoteOn".to_string() },
            EventData::NoteOff { .. } => "NoteOff".to_string(),
            EventData::ControlChange { .. } => "ControlChange".to_string(),
            EventData::ProgramChange { .. } => "ProgramChange".to_string(),
            EventData::PitchBend { .. } => "PitchBend".to_string(),
            EventData::Tempo { .. } => "Tempo".to_string(),
            EventData::TimeSignature { .. } => "TimeSignature".to_string(),
            EventData::KeySignature { .. } => "KeySignature".to_string(),
            EventData::TrackName { .. } => "TrackName".to_string(),
            EventData::EndOfTrack => "EndOfTrack".to_string(),
            EventData::OtherMeta { .. } => "OtherMeta".to_string(),
            EventData::SysEx { .. } => "SysEx".to_string(),
            EventData::OtherChannel { .. } => "OtherChannel".to_string(),
        }
    }

    #[getter]
    fn tick(&self) -> u32 { self.inner.absolute_tick }

    #[setter]
    fn set_tick(&mut self, t: u32) { self.inner.absolute_tick = t; }

    #[getter]
    fn channel(&self) -> Option<u8> {
        match &self.inner.data {
            EventData::NoteOn { ch, .. } | EventData::NoteOff { ch, .. } | EventData::ControlChange { ch, .. } | EventData::ProgramChange { ch, .. } | EventData::PitchBend { ch, .. } => Some(*ch),
            _ => None,
        }
    }

    #[getter]
    fn pitch(&self) -> Option<u8> {
        match &self.inner.data {
            EventData::NoteOn { pitch, .. } | EventData::NoteOff { pitch, .. } => Some(*pitch),
            _ => None,
        }
    }

    #[getter]
    fn velocity(&self) -> Option<u8> {
        match &self.inner.data {
            EventData::NoteOn { vel, .. } | EventData::NoteOff { vel, .. } => Some(*vel),
            _ => None,
        }
    }

    #[getter]
    fn tempo(&self) -> Option<u32> {
        if let EventData::Tempo { usec_per_quarter } = &self.inner.data { Some(*usec_per_quarter) } else { None }
    }

    #[getter]
    fn numerator(&self) -> Option<u8> {
        if let EventData::TimeSignature { num, .. } = &self.inner.data { Some(*num) } else { None }
    }

    #[getter]
    fn denominator(&self) -> Option<u32> {
        if let EventData::TimeSignature { den, .. } = &self.inner.data { Some(2u32.pow(*den as u32)) } else { None }
    }

    #[getter]
    fn controller(&self) -> Option<u8> {
        if let EventData::ControlChange { cc, .. } = &self.inner.data { Some(*cc) } else { None }
    }

    #[getter]
    fn value(&self) -> Option<u16> {
        match &self.inner.data {
            EventData::ControlChange { val, .. } => Some(*val as u16),
            EventData::PitchBend { val, .. } => Some(*val),
            _ => None,
        }
    }

    #[getter]
    fn program(&self) -> Option<u8> {
        if let EventData::ProgramChange { prog, .. } = &self.inner.data { Some(*prog) } else { None }
    }

    #[getter]
    fn bend(&self) -> Option<u16> {
        if let EventData::PitchBend { val, .. } = &self.inner.data { Some(*val) } else { None }
    }

    #[getter]
    fn key(&self) -> Option<i8> {
        if let EventData::KeySignature { sf, .. } = &self.inner.data { Some(*sf) } else { None }
    }

    #[getter]
    fn scale(&self) -> Option<u8> {
        if let EventData::KeySignature { mi, .. } = &self.inner.data { Some(*mi) } else { None }
    }
}

#[pyclass]
pub struct Track {
    pub name: Option<String>,
    pub events: Vec<Event>,
}

#[pymethods]
impl Track {
    #[new]
    fn new(name: Option<String>) -> Self {
        Track { name, events: Vec::new() }
    }

    #[getter]
    fn name(&self) -> Option<String> { self.name.clone() }

    #[setter]
    fn set_name(&mut self, name: Option<String>) { self.name = name; }

    #[getter]
    fn events(&self) -> Vec<Event> { self.events.clone() }

    #[setter]
    fn set_events(&mut self, events: Vec<Event>) { self.events = events; }

    fn add_event(&mut self, ev: Event) {
        self.events.push(ev);
    }

    fn remove_event(&mut self, index: usize) {
        if index < self.events.len() {
            self.events.remove(index);
        }
    }

    fn sort_events(&mut self) {
        let mut core_track = midi_core::Track {
            name: self.name.clone(),
            events: self.events.iter().map(|e| e.inner.clone()).collect(),
        };
        core_track.sort_events();
        self.events = core_track.events.into_iter().map(|e| Event { inner: e }).collect();
    }
}

#[pyclass]
pub struct Score {
    format_type: u16,
    ticks_per_quarter: u16,
    tracks_storage: Vec<Py<Track>>,
}

impl Score {
    fn to_core(&self, py: Python<'_>) -> midi_core::Score {
        midi_core::Score {
            format: self.format_type,
            ticks_per_quarter: self.ticks_per_quarter,
            tracks: self.tracks_storage.iter().map(|py_track| {
                let track = py_track.borrow(py);
                midi_core::Track {
                    name: track.name.clone(),
                    events: track.events.iter().map(|e| e.inner.clone()).collect(),
                }
            }).collect(),
        }
    }

    fn from_core(&mut self, py: Python<'_>, core: midi_core::Score) -> PyResult<()> {
        self.format_type = core.format;
        self.ticks_per_quarter = core.ticks_per_quarter;
        self.tracks_storage = core.tracks.into_iter().map(|t| {
            Py::new(py, Track {
                name: t.name,
                events: t.events.into_iter().map(|inner| Event { inner }).collect(),
            })
        }).collect::<PyResult<Vec<_>>>()?;
        Ok(())
    }
}

#[pymethods]
impl Score {
    #[new]
    fn new(format_type: u16, ticks_per_quarter: u16) -> Self {
        Score { format_type, ticks_per_quarter, tracks_storage: Vec::new() }
    }

    #[getter]
    fn format_type(&self) -> u16 { self.format_type }

    #[setter]
    fn set_format_type(&mut self, v: u16) { self.format_type = v; }

    #[getter]
    fn ticks_per_quarter(&self) -> u16 { self.ticks_per_quarter }

    #[setter]
    fn set_ticks_per_quarter(&mut self, v: u16) { self.ticks_per_quarter = v; }

    #[getter]
    fn tracks(&self) -> Vec<Py<Track>> {
        self.tracks_storage.clone()
    }

    #[setter]
    fn set_tracks(&mut self, tracks: Vec<Py<Track>>) {
        self.tracks_storage = tracks;
    }

    #[staticmethod]
    fn from_file(py: Python<'_>, path: &str) -> PyResult<Self> {
        let core_score = midi_core::Score::from_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let tracks_storage = core_score.tracks.into_iter().map(|trk| {
            Py::new(py, Track {
                name: trk.name,
                events: trk.events.into_iter().map(|inner| Event { inner }).collect(),
            })
        }).collect::<PyResult<Vec<_>>>()?;

        Ok(Score {
            format_type: core_score.format,
            ticks_per_quarter: core_score.ticks_per_quarter,
            tracks_storage,
        })
    }

    fn to_file(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        let core_score = self.to_core(py);
        core_score.to_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    fn merge_tracks(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut core_score = self.to_core(py);
        core_score.merge_tracks();
        self.from_core(py, core_score)
    }

    fn merge_tracks_by_program(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut core_score = self.to_core(py);
        core_score.merge_tracks_by_program();
        self.from_core(py, core_score)
    }
}

#[pymodule]
fn midi_toolkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Score>()?;
    m.add_class::<Track>()?;
    m.add_class::<Event>()?;
    Ok(())
}
