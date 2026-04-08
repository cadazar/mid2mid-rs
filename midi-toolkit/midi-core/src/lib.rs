use std::io::{Read, Write, Cursor};
use std::fs::File;

#[derive(Debug, Clone, PartialEq)]
pub enum EventData {
    NoteOff { ch: u8, pitch: u8, vel: u8 },
    NoteOn { ch: u8, pitch: u8, vel: u8 },
    ControlChange { ch: u8, cc: u8, val: u8 },
    ProgramChange { ch: u8, prog: u8 },
    PitchBend { ch: u8, val: u16 },
    Tempo { usec_per_quarter: u32 },
    TimeSignature { num: u8, den: u8, clocks_per_click: u8, n_32nd_notes: u8 },
    KeySignature { sf: i8, mi: u8 },
    TrackName { text: Vec<u8> },
    EndOfTrack,
    OtherMeta { type_: u8, data: Vec<u8> },
    SysEx { data: Vec<u8> },
    OtherChannel { status: u8, data: Vec<u8> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub delta_tick: u32,
    pub absolute_tick: u32,
    pub data: EventData,
    pub has_status_byte: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Track {
    pub name: Option<String>,
    pub events: Vec<Event>,
}

impl Track {
    pub fn sort_events(&mut self) {
        self.events.sort_by(|a, b| {
            if a.absolute_tick != b.absolute_tick {
                return a.absolute_tick.cmp(&b.absolute_tick);
            }
            let a_is_eot = matches!(a.data, EventData::EndOfTrack);
            let b_is_eot = matches!(b.data, EventData::EndOfTrack);
            if a_is_eot && !b_is_eot { return std::cmp::Ordering::Greater; }
            if !a_is_eot && b_is_eot { return std::cmp::Ordering::Less; }
            
            let is_note_off = |e: &EventData| matches!(e, EventData::NoteOff { .. });
            let is_note_on = |e: &EventData| matches!(e, EventData::NoteOn { .. });
            
            if is_note_off(&a.data) && is_note_on(&b.data) {
                if let (EventData::NoteOff { pitch: p1, ch: c1, .. }, EventData::NoteOn { pitch: p2, ch: c2, .. }) = (&a.data, &b.data) {
                    if p1 == p2 && c1 == c2 {
                        return std::cmp::Ordering::Less;
                    }
                }
            } else if is_note_on(&a.data) && is_note_off(&b.data) {
                if let (EventData::NoteOn { pitch: p1, ch: c1, .. }, EventData::NoteOff { pitch: p2, ch: c2, .. }) = (&a.data, &b.data) {
                    if p1 == p2 && c1 == c2 {
                        return std::cmp::Ordering::Greater;
                    }
                }
            }
            std::cmp::Ordering::Equal
        });
        
        let mut last_tick = 0;
        for ev in &mut self.events {
            ev.delta_tick = ev.absolute_tick - last_tick;
            last_tick = ev.absolute_tick;
            ev.has_status_byte = true; // after sorting, always write status bytes to be safe
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    pub format: u16,
    pub ticks_per_quarter: u16,
    pub tracks: Vec<Track>,
}

fn read_vlq<R: Read>(reader: &mut R) -> std::io::Result<u32> {
    let mut value = 0u32;
    let mut buf = [0u8; 1];
    loop {
        reader.read_exact(&mut buf)?;
        let byte = buf[0];
        value = (value << 7) | ((byte & 0x7F) as u32);
        if byte & 0x80 == 0 {
            break;
        }
    }
    Ok(value)
}

fn write_vlq<W: Write>(writer: &mut W, mut value: u32) -> std::io::Result<()> {
    let mut buffer = [0u8; 4];
    let mut i = 0;
    buffer[i] = (value & 0x7F) as u8;
    while value > 0x7F {
        value >>= 7;
        i += 1;
        buffer[i] = ((value & 0x7F) | 0x80) as u8;
    }
    while i > 0 {
        writer.write_all(&[buffer[i]])?;
        i -= 1;
    }
    writer.write_all(&[buffer[0]])?;
    Ok(())
}

fn read_u16_be<R: Read>(reader: &mut R) -> std::io::Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_be_bytes(buf))
}

fn read_u32_be<R: Read>(reader: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn write_u16_be<W: Write>(writer: &mut W, val: u16) -> std::io::Result<()> {
    writer.write_all(&val.to_be_bytes())
}

fn write_u32_be<W: Write>(writer: &mut W, val: u32) -> std::io::Result<()> {
    writer.write_all(&val.to_be_bytes())
}

impl Score {
    pub fn read<R: Read>(mut reader: R) -> std::io::Result<Self> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != b"MThd" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid MIDI header"));
        }
        let header_len = read_u32_be(&mut reader)?;
        if header_len < 6 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid MIDI header length"));
        }
        let format = read_u16_be(&mut reader)?;
        let n_tracks = read_u16_be(&mut reader)?;
        let ticks_per_quarter = read_u16_be(&mut reader)?;
        
        if header_len > 6 {
            let mut skip = vec![0u8; (header_len - 6) as usize];
            reader.read_exact(&mut skip)?;
        }

        let mut tracks = Vec::new();
        for _ in 0..n_tracks {
            reader.read_exact(&mut magic)?;
            if &magic != b"MTrk" {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid track header"));
            }
            let track_len = read_u32_be(&mut reader)?;
            let mut track_data = vec![0u8; track_len as usize];
            reader.read_exact(&mut track_data)?;
            
            let mut cursor = Cursor::new(track_data);
            let mut absolute_tick = 0;
            let mut running_status = 0u8;
            let mut events = Vec::new();
            let mut track_name = None;

            while (cursor.position() as usize) < track_len as usize {
                let delta = read_vlq(&mut cursor)?;
                absolute_tick += delta;
                
                let mut status_buf = [0u8; 1];
                cursor.read_exact(&mut status_buf)?;
                let mut status = status_buf[0];
                let mut has_status_byte = true;
                
                if status < 0x80 {
                    status = running_status;
                    has_status_byte = false;
                    cursor.set_position(cursor.position() - 1);
                } else {
                    if status < 0xF0 {
                        running_status = status;
                    }
                }
                
                let data = if status == 0xFF {
                    let mut type_buf = [0u8; 1];
                    cursor.read_exact(&mut type_buf)?;
                    let type_ = type_buf[0];
                    let len = read_vlq(&mut cursor)?;
                    let mut meta_data = vec![0u8; len as usize];
                    cursor.read_exact(&mut meta_data)?;
                    
                    match type_ {
                        0x03 => {
                            if track_name.is_none() {
                                track_name = Some(String::from_utf8_lossy(&meta_data).into_owned());
                            }
                            EventData::TrackName { text: meta_data }
                        },
                        0x2F => EventData::EndOfTrack,
                        0x51 => {
                            let usec = ((meta_data[0] as u32) << 16) | ((meta_data[1] as u32) << 8) | (meta_data[2] as u32);
                            EventData::Tempo { usec_per_quarter: usec }
                        },
                        0x58 => {
                            EventData::TimeSignature {
                                num: meta_data[0],
                                den: meta_data[1],
                                clocks_per_click: meta_data[2],
                                n_32nd_notes: meta_data[3],
                            }
                        },
                        0x59 => {
                            EventData::KeySignature {
                                sf: meta_data[0] as i8,
                                mi: meta_data[1],
                            }
                        },
                        _ => EventData::OtherMeta { type_, data: meta_data },
                    }
                } else if status == 0xF0 || status == 0xF7 {
                    let len = read_vlq(&mut cursor)?;
                    let mut sysex_data = vec![0u8; len as usize];
                    cursor.read_exact(&mut sysex_data)?;
                    EventData::SysEx { data: sysex_data }
                } else {
                    let ch = status & 0x0F;
                    let msg = status & 0xF0;
                    match msg {
                        0x80 => {
                            let mut buf = [0u8; 2]; cursor.read_exact(&mut buf)?;
                            EventData::NoteOff { ch, pitch: buf[0], vel: buf[1] }
                        },
                        0x90 => {
                            let mut buf = [0u8; 2]; cursor.read_exact(&mut buf)?;
                            EventData::NoteOn { ch, pitch: buf[0], vel: buf[1] }
                        },
                        0xA0 | 0xB0 => {
                            let mut buf = [0u8; 2]; cursor.read_exact(&mut buf)?;
                            if msg == 0xB0 {
                                EventData::ControlChange { ch, cc: buf[0], val: buf[1] }
                            } else {
                                EventData::OtherChannel { status, data: buf.to_vec() }
                            }
                        },
                        0xC0 | 0xD0 => {
                            let mut buf = [0u8; 1]; cursor.read_exact(&mut buf)?;
                            if msg == 0xC0 {
                                EventData::ProgramChange { ch, prog: buf[0] }
                            } else {
                                EventData::OtherChannel { status, data: buf.to_vec() }
                            }
                        },
                        0xE0 => {
                            let mut buf = [0u8; 2]; cursor.read_exact(&mut buf)?;
                            let val = (buf[0] as u16) | ((buf[1] as u16) << 7);
                            EventData::PitchBend { ch, val }
                        },
                        _ => {
                            EventData::OtherChannel { status, data: vec![] }
                        }
                    }
                };
                
                events.push(Event { delta_tick: delta, absolute_tick, data, has_status_byte });
            }
            
            tracks.push(Track { name: track_name, events });
        }
        
        Ok(Score { format, ticks_per_quarter, tracks })
    }

    pub fn write<W: Write>(&self, mut writer: W) -> std::io::Result<()> {
        writer.write_all(b"MThd")?;
        write_u32_be(&mut writer, 6)?;
        write_u16_be(&mut writer, self.format)?;
        write_u16_be(&mut writer, self.tracks.len() as u16)?;
        write_u16_be(&mut writer, self.ticks_per_quarter)?;

        for track in &self.tracks {
            writer.write_all(b"MTrk")?;
            let mut track_buf = Vec::new();
            
            let mut trk_clone = track.clone();
            
            let mut last_tick = 0;
            for ev in &mut trk_clone.events {
                ev.delta_tick = ev.absolute_tick - last_tick;
                last_tick = ev.absolute_tick;
            }
            
            let mut running_status = 0u8;
            for ev in &trk_clone.events {
                write_vlq(&mut track_buf, ev.delta_tick)?;
                
                match &ev.data {
                    EventData::NoteOff { ch, pitch, vel } => {
                        let st = 0x80 | (ch & 0x0F);
                        if ev.has_status_byte || st != running_status {
                            track_buf.push(st);
                            running_status = st;
                        }
                        track_buf.push(*pitch);
                        track_buf.push(*vel);
                    },
                    EventData::NoteOn { ch, pitch, vel } => {
                        let st = 0x90 | (ch & 0x0F);
                        if ev.has_status_byte || st != running_status {
                            track_buf.push(st);
                            running_status = st;
                        }
                        track_buf.push(*pitch);
                        track_buf.push(*vel);
                    },
                    EventData::ControlChange { ch, cc, val } => {
                        let st = 0xB0 | (ch & 0x0F);
                        if ev.has_status_byte || st != running_status {
                            track_buf.push(st);
                            running_status = st;
                        }
                        track_buf.push(*cc);
                        track_buf.push(*val);
                    },
                    EventData::ProgramChange { ch, prog } => {
                        let st = 0xC0 | (ch & 0x0F);
                        if ev.has_status_byte || st != running_status {
                            track_buf.push(st);
                            running_status = st;
                        }
                        track_buf.push(*prog);
                    },
                    EventData::PitchBend { ch, val } => {
                        let st = 0xE0 | (ch & 0x0F);
                        if ev.has_status_byte || st != running_status {
                            track_buf.push(st);
                            running_status = st;
                        }
                        track_buf.push((val & 0x7F) as u8);
                        track_buf.push(((val >> 7) & 0x7F) as u8);
                    },
                    EventData::Tempo { usec_per_quarter } => {
                        track_buf.push(0xFF);
                        track_buf.push(0x51);
                        write_vlq(&mut track_buf, 3)?;
                        track_buf.push(((usec_per_quarter >> 16) & 0xFF) as u8);
                        track_buf.push(((usec_per_quarter >> 8) & 0xFF) as u8);
                        track_buf.push((usec_per_quarter & 0xFF) as u8);
                    },
                    EventData::TimeSignature { num, den, clocks_per_click, n_32nd_notes } => {
                        track_buf.push(0xFF);
                        track_buf.push(0x58);
                        write_vlq(&mut track_buf, 4)?;
                        track_buf.push(*num);
                        track_buf.push(*den);
                        track_buf.push(*clocks_per_click);
                        track_buf.push(*n_32nd_notes);
                    },
                    EventData::KeySignature { sf, mi } => {
                        track_buf.push(0xFF);
                        track_buf.push(0x59);
                        write_vlq(&mut track_buf, 2)?;
                        track_buf.push(*sf as u8);
                        track_buf.push(*mi);
                    },
                    EventData::TrackName { text } => {
                        track_buf.push(0xFF);
                        track_buf.push(0x03);
                        write_vlq(&mut track_buf, text.len() as u32)?;
                        track_buf.extend_from_slice(text);
                    },
                    EventData::EndOfTrack => {
                        track_buf.push(0xFF);
                        track_buf.push(0x2F);
                        write_vlq(&mut track_buf, 0)?;
                    },
                    EventData::OtherMeta { type_, data } => {
                        track_buf.push(0xFF);
                        track_buf.push(*type_);
                        write_vlq(&mut track_buf, data.len() as u32)?;
                        track_buf.extend_from_slice(data);
                    },
                    EventData::SysEx { data } => {
                        track_buf.push(0xF0);
                        write_vlq(&mut track_buf, data.len() as u32)?;
                        track_buf.extend_from_slice(data);
                    },
                    EventData::OtherChannel { status, data } => {
                        if ev.has_status_byte || *status != running_status {
                            track_buf.push(*status);
                            running_status = *status;
                        }
                        track_buf.extend_from_slice(data);
                    }
                }
            }
            write_u32_be(&mut writer, track_buf.len() as u32)?;
            writer.write_all(&track_buf)?;
        }
        Ok(())
    }

    pub fn from_file(path: &str) -> std::io::Result<Self> {
        let f = File::open(path)?;
        Self::read(f)
    }

    pub fn to_file(&self, path: &str) -> std::io::Result<()> {
        let f = File::create(path)?;
        self.write(f)
    }

    pub fn merge_tracks(&mut self) {
        if self.tracks.is_empty() { return; }
        let mut merged_events = Vec::new();
        for track in &self.tracks {
            merged_events.extend(track.events.clone());
        }
        let mut merged = Track { name: Some("Merged".to_string()), events: merged_events };
        merged.sort_events();
        self.tracks = vec![merged];
        self.format = 0;
    }

    pub fn merge_tracks_by_program(&mut self) {
        use std::collections::HashMap;
        let mut groups: HashMap<u8, Vec<Event>> = HashMap::new();
        
        for track in &self.tracks {
            let mut current_prog = 0;
            for ev in &track.events {
                if let EventData::ProgramChange { prog, .. } = &ev.data {
                    current_prog = *prog;
                    break;
                }
            }
            
            groups.entry(current_prog).or_default().extend(track.events.clone());
        }
        
        let mut new_tracks = Vec::new();
        let mut progs: Vec<u8> = groups.keys().cloned().collect();
        progs.sort();
        
        for p in progs {
            let mut trk = Track { name: Some(format!("Program {}", p)), events: groups.remove(&p).unwrap() };
            trk.sort_events();
            new_tracks.push(trk);
        }
        
        self.tracks = new_tracks;
        if self.tracks.len() <= 1 {
            self.format = 0;
        } else {
            self.format = 1;
        }
    }
}
