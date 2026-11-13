//! Input history for dilated convolutions.
//!
//! Every `Conv1D` needs the last `(kernel_size - 1) * dilation` frames of
//! its input in addition to the current block. There are two ways to keep
//! them, one for each way a convolution's lanes can run
//! ([`crate::kernels`]), and what differs is which axis is contiguous.
//!
//! # Frames together: [`History`]
//!
//! A convolution whose lanes are its output channels reads a frame at a
//! time, all channels of it, so a frame is contiguous. The C++ reference
//! keeps that in a `RingBuffer` that, when the write pointer runs out of
//! room, bulk-copies `max_lookback` columns back to the start.
//!
//! This does not copy. A dilated convolution never needs the whole lookback
//! contiguous. It needs `num_frames` contiguous at each of `kernel_size`
//! fixed offsets, so the read pointers wrap instead, and each read splits
//! into at most two runs. That is bit-exact against the reference by
//! construction, because a read is a read: the values returned are the same
//! ones, reached by different arithmetic on the index. `tests/reference.rs`
//! holds the whole-model consequence against the reference's own output.
//!
//! The storage is `max_lookback + max_buffer` columns per convolution, 40%
//! less than the reference's `2 * max_lookback + max_buffer`, and no block
//! is more expensive than any other, since none triggers a copy. The copy
//! is not where a model's time goes, since removing it measures within
//! noise, so the design is about storage and worst case, not speed.
//!
//! Per-layer buffers would scatter the working set, so every such history
//! in a model is backed by one [`Arena`]: a single contiguous allocation,
//! laid out in execution order.
//!
//! # Channels apart: [`Planes`]
//!
//! A convolution whose lanes are consecutive frames reads four frames of
//! one channel at a time, so a channel is contiguous: one row each. A read
//! that wrapped would split a register, so this one is linear and does
//! copy, the lookback back to the start once the rows run out. A row is as
//! long as the reference's ring, `2 * max_lookback + max_buffer`, so the
//! copy is at most a frame for each frame taken in; and it is never less
//! than [`ROOM`] frames past its block, so a history a few frames long is
//! not copied every block either.

/// One contiguous allocation holding every convolution's input history.
#[derive(Debug, Clone, Default)]
pub(crate) struct Arena {
    data: Vec<f32>,
}

impl Arena {
    /// An empty arena.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Drop all reservations. Call before re-reserving for a new buffer size.
    pub(crate) fn clear(&mut self) {
        self.data.clear();
    }

    /// Reserve `len` zeroed floats and return the offset of the reservation.
    pub(crate) fn alloc(&mut self, len: usize) -> usize {
        let off = self.data.len();
        self.data.resize(off + len, 0.0);
        off
    }

    /// Total size in floats.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn slice(&self, off: usize, len: usize) -> &[f32] {
        &self.data[off..off + len]
    }

    #[inline]
    fn slice_mut(&mut self, off: usize, len: usize) -> &mut [f32] {
        &mut self.data[off..off + len]
    }
}

/// A single convolution's input history, backed by a slice of an [`Arena`].
#[derive(Debug, Clone)]
pub(crate) struct History {
    rows: usize,
    /// Number of columns of storage.
    cols: usize,
    /// `(kernel_size - 1) * dilation`.
    max_lookback: usize,
    /// Largest block this history was sized for.
    max_buffer: usize,
    /// Column index of the next write.
    write_pos: usize,
    /// Offset into the arena, in floats.
    offset: usize,
}

impl History {
    /// Reserve storage for `rows` channels, `max_lookback` frames of history
    /// and blocks of up to `max_buffer` frames.
    ///
    /// `max_lookback + max_buffer` columns is enough because nothing is ever
    /// copied: after writing `n` columns there are still
    /// `cols - n >= max_lookback` columns of intact history behind the write
    /// pointer. The reference reserves `2 * max_lookback + max_buffer`, the
    /// doubled lookback being what keeps its rewind copy non-overlapping.
    pub(crate) fn reserve(
        arena: &mut Arena,
        rows: usize,
        max_lookback: usize,
        max_buffer: usize,
    ) -> Self {
        let cols = max_lookback + max_buffer;
        let offset = arena.alloc(rows * cols);
        Self {
            rows,
            cols,
            max_lookback,
            max_buffer,
            write_pos: 0,
            offset,
        }
    }

    /// Write `n` frames from `src` (a column-major `(rows, >= n)` buffer) at
    /// the write pointer, wrapping around the end if needed.
    pub(crate) fn write(&mut self, arena: &mut Arena, src: &[f32], n: usize) {
        debug_assert!(
            n <= self.max_buffer,
            "block of {n} exceeds max buffer {}",
            self.max_buffer
        );
        debug_assert!(src.len() >= self.rows * n);
        let first = (self.cols - self.write_pos).min(n);
        let start = self.offset + self.write_pos * self.rows;
        arena
            .slice_mut(start, self.rows * first)
            .copy_from_slice(&src[..self.rows * first]);
        if first < n {
            let rest = n - first;
            arena
                .slice_mut(self.offset, self.rows * rest)
                .copy_from_slice(&src[self.rows * first..self.rows * n]);
        }
    }

    /// Advance the write pointer by `n`, after the block has been read.
    #[inline]
    pub(crate) fn advance(&mut self, n: usize) {
        self.write_pos += n;
        self.write_pos %= self.cols;
    }

    /// The runs of storage covering `n` frames ending `lookback` frames before
    /// the write pointer.
    ///
    /// Returns `[(first_col, count), (0, count)]`; the second run has
    /// `count == 0` unless the window wraps the end of the storage.
    #[inline]
    pub(crate) fn read_runs(&self, n: usize, lookback: usize) -> [(usize, usize); 2] {
        debug_assert!(lookback <= self.max_lookback);
        debug_assert!(n <= self.max_buffer);
        let start = (self.write_pos + self.cols - lookback) % self.cols;
        let first = (self.cols - start).min(n);
        [(start, first), (0, n - first)]
    }

    /// Borrow `count` columns starting at storage column `col`.
    #[inline]
    pub(crate) fn run<'a>(&self, arena: &'a Arena, col: usize, count: usize) -> &'a [f32] {
        arena.slice(self.offset + col * self.rows, self.rows * count)
    }

    /// Copy the most recently written column into `dst`.
    ///
    /// Used by the prewarm cache: after prewarming on silence the model is in
    /// a steady state, and every convolution's history is a constant column.
    pub(crate) fn cache_last_written(&self, arena: &Arena, dst: &mut [f32]) {
        debug_assert_eq!(dst.len(), self.rows);
        let col = (self.write_pos + self.cols - 1) % self.cols;
        dst.copy_from_slice(self.run(arena, col, 1));
    }

    /// Fill every column with `sample` and restore the initial write position.
    ///
    /// The reference's `RingBuffer::FillWithSample`. Since every column ends up
    /// identical, the write position only has to be somewhere legal, which
    /// is why this can restart at 0 where the reference restarts at
    /// `max_lookback` and still agree with it.
    pub(crate) fn fill_with_sample(&mut self, arena: &mut Arena, sample: &[f32]) {
        debug_assert_eq!(sample.len(), self.rows);
        let rows = self.rows;
        let base = self.offset;
        for c in 0..self.cols {
            arena
                .slice_mut(base + c * rows, rows)
                .copy_from_slice(sample);
        }
        self.write_pos = 0;
    }
}

/// Frames a [`Planes`] takes in between two rewinds, at the least. A rewind
/// costs a call per row however little it copies, so a history a few frames
/// long is given this much room rather than its own length again. Measured
/// on the three-channel A2 capture, most of whose layers reach back under a
/// hundred frames: 256 and 4096 are both slower at blocks of 16.
const ROOM: usize = 1024;

/// A convolution's input history with a row per channel, each row's frames
/// contiguous: linear, with a rewind when the rows run out, so that every
/// read is one run.
#[derive(Debug, Clone)]
pub(crate) struct Planes {
    /// `rows` rows of `cap` frames, one after another.
    data: Vec<f32>,
    rows: usize,
    cap: usize,
    /// Frames kept behind the write position: the lookback, and at least
    /// one, so that there is always a last frame to cache.
    keep: usize,
    /// The frame the current block starts at; at least `keep`.
    pos: usize,
}

impl Planes {
    /// Storage for `rows` channels, `lookback` frames of history and blocks
    /// of up to `max_block` frames, cleared.
    ///
    /// A row holds the lookback, a block, and room for as many frames again
    /// as the lookback before it is rewound, [`ROOM`] at the least: what a
    /// rewind copies is then no more than what came in since the last one,
    /// and the storage no more than the reference's ring plus that floor.
    pub(crate) fn new(rows: usize, lookback: usize, max_block: usize) -> Self {
        let keep = lookback.max(1);
        let cap = keep + max_block + keep.max(ROOM);
        Self {
            data: vec![0.0; rows * cap],
            rows,
            cap,
            keep,
            pos: keep,
        }
    }

    /// Make room for a block of `n` frames at [`Planes::pos`]: rewind when
    /// the end is near, taking the kept frames along.
    #[inline(always)]
    pub(crate) fn begin(&mut self, n: usize) {
        debug_assert!(self.keep + n <= self.cap, "block of {n} is too long");
        if self.pos + n > self.cap {
            let (keep, pos) = (self.keep, self.pos);
            for row in self.data.chunks_exact_mut(self.cap) {
                row.copy_within(pos - keep..pos, 0);
            }
            self.pos = keep;
        }
    }

    /// The frame the current block starts at, in every row.
    #[inline(always)]
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    /// The distance in floats from one row to the next.
    #[inline(always)]
    pub(crate) fn stride(&self) -> usize {
        self.cap
    }

    /// Every row, one after another.
    #[inline(always)]
    pub(crate) fn data(&self) -> &[f32] {
        &self.data
    }

    /// The rows, in channel order, each whole.
    #[inline(always)]
    pub(crate) fn rows_mut(&mut self) -> std::slice::ChunksExactMut<'_, f32> {
        self.data.chunks_exact_mut(self.cap)
    }

    /// Move the write position past a block of `n` frames, after it has
    /// been written and read.
    #[inline(always)]
    pub(crate) fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    /// Back to all zeros.
    pub(crate) fn clear(&mut self) {
        self.data.fill(0.0);
        self.pos = self.keep;
    }

    /// Copy the most recently written frame into `dst`, a float per row.
    pub(crate) fn last_written(&self, dst: &mut [f32]) {
        debug_assert_eq!(dst.len(), self.rows);
        for (d, row) in dst.iter_mut().zip(self.data.chunks_exact(self.cap)) {
            *d = row[self.pos - 1];
        }
    }

    /// Fill every frame with `sample`, a float per row, as
    /// [`History::fill_with_sample`] does.
    pub(crate) fn fill_with_sample(&mut self, sample: &[f32]) {
        debug_assert_eq!(sample.len(), self.rows);
        for (row, s) in self.data.chunks_exact_mut(self.cap).zip(sample) {
            row.fill(*s);
        }
        self.pos = self.keep;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ring is checked against a naive history that simply keeps every
    /// frame ever written, over ragged block schedules and every lookback.
    ///
    /// The oracle is the definition, not a second ring buffer, because two
    /// rings can agree with each other and both be wrong: the window ending `lookback`
    /// frames before the write pointer is the last `n` frames of everything
    /// written, offset by `lookback`.
    #[test]
    fn reads_match_a_naive_full_history() {
        const ROWS: usize = 3;
        for &(lookback, max_buf) in &[
            (0usize, 8usize),
            (1, 8),
            (7, 8),
            (16, 64),
            (239, 64),
            (5, 1),
            (512, 128),
        ] {
            let mut arena = Arena::new();
            let mut h = History::reserve(&mut arena, ROWS, lookback, max_buf);
            // Every frame ever written, column-major, starting from the zeroed
            // history the ring begins with.
            let mut all = vec![0.0f32; ROWS * (lookback + max_buf)];

            let schedule = [1usize, 7, 3, max_buf, 2, max_buf, 5, 1, max_buf.min(4)];
            let mut counter = 0.0f32;
            for round in 0..40 {
                let n = schedule[round % schedule.len()].clamp(1, max_buf);
                let mut src = vec![0.0f32; ROWS * n];
                for v in src.iter_mut() {
                    counter += 1.0;
                    *v = counter;
                }
                h.write(&mut arena, &src, n);
                all.extend_from_slice(&src);

                for lb in [0, lookback / 2, lookback] {
                    let mut got = Vec::new();
                    for (c, k) in h.read_runs(n, lb) {
                        if k > 0 {
                            got.extend_from_slice(h.run(&arena, c, k));
                        }
                    }
                    // The same window, taken straight from the full history.
                    let end = all.len() / ROWS - lb;
                    let want = &all[(end - n) * ROWS..end * ROWS];
                    assert_eq!(
                        got, want,
                        "lookback={lookback} max_buf={max_buf} round={round} lb={lb}"
                    );
                }
                h.advance(n);
            }
        }
    }

    /// Storage is `max_lookback + max_buffer` columns, not the reference's
    /// `2 * max_lookback + max_buffer`.
    #[test]
    fn storage_is_lookback_plus_buffer() {
        let mut arena = Arena::new();
        let _ = History::reserve(&mut arena, 8, 239, 64);
        assert_eq!(arena.len(), 8 * (239 + 64));
    }

    /// The planar history against the same naive one: whatever the block
    /// schedule and however many rewinds it takes, the frames behind the
    /// block are the ones that were written.
    #[test]
    fn planes_keep_the_lookback_through_every_rewind() {
        const ROWS: usize = 3;
        for &(lookback, max_block) in &[(0usize, 8usize), (1, 8), (7, 8), (239, 64), (5, 1)] {
            let mut p = Planes::new(ROWS, lookback, max_block);
            // Every frame ever written, a `Vec` per row, from a zeroed past.
            let mut all = vec![vec![0.0f32; lookback.max(1)]; ROWS];
            let schedule = [
                1usize,
                7,
                3,
                max_block,
                2,
                max_block,
                5,
                1,
                max_block.min(4),
            ];
            let mut counter = 0.0f32;
            let mut rewinds = 0;
            for round in 0..8000 {
                let n = schedule[round % schedule.len()].clamp(1, max_block);
                let before = p.pos();
                p.begin(n);
                rewinds += usize::from(p.pos() < before);
                let pos = p.pos();
                for (row, all) in p.rows_mut().zip(all.iter_mut()) {
                    for v in &mut row[pos..pos + n] {
                        counter += 1.0;
                        *v = counter;
                        all.push(counter);
                    }
                }
                for (c, all) in all.iter().enumerate() {
                    let row = &p.data()[c * p.stride()..(c + 1) * p.stride()];
                    let want = &all[all.len() - n - lookback..];
                    assert_eq!(
                        &row[pos - lookback..pos + n],
                        want,
                        "lookback={lookback} max_block={max_block} round={round} row={c}"
                    );
                }
                p.advance(n);
                let mut last = [0.0f32; ROWS];
                p.last_written(&mut last);
                for (c, all) in all.iter().enumerate() {
                    assert_eq!(last[c], *all.last().unwrap());
                }
            }
            assert!(rewinds > 2, "lookback={lookback}: the rows never ran out");
        }
    }

    /// A long history is stored as the reference stores it, twice the
    /// lookback and a block; a short one has `ROOM` frames instead of its
    /// own length again.
    #[test]
    fn planes_are_twice_the_lookback_or_the_room_and_a_block() {
        let long = Planes::new(8, 5000, 64);
        assert_eq!(long.data().len(), 8 * (2 * 5000 + 64));
        let short = Planes::new(8, 15, 64);
        assert_eq!(short.data().len(), 8 * (15 + ROOM + 64));
        // No lookback at all still keeps the one frame a cache can read.
        let none = Planes::new(2, 0, 4);
        assert_eq!(none.data().len(), 2 * (1 + ROOM + 4));
    }

    #[test]
    fn planes_filled_with_a_sample_hold_it_at_every_lookback() {
        let mut p = Planes::new(2, 10, 4);
        p.fill_with_sample(&[0.25, -0.5]);
        p.begin(4);
        let pos = p.pos();
        for (c, want) in [0.25f32, -0.5].into_iter().enumerate() {
            let row = &p.data()[c * p.stride()..(c + 1) * p.stride()];
            assert!(row[pos - 10..pos].iter().all(|v| *v == want));
        }
        let mut last = [0.0f32; 2];
        p.last_written(&mut last);
        assert_eq!(last, [0.25, -0.5]);
        // A history nothing has been written to has a last frame too: zeros.
        let fresh = Planes::new(2, 0, 4);
        fresh.last_written(&mut last);
        assert_eq!(last, [0.0, 0.0]);
    }

    #[test]
    fn fill_with_sample_is_seen_at_every_lookback() {
        let mut arena = Arena::new();
        let mut h = History::reserve(&mut arena, 2, 10, 4);
        h.fill_with_sample(&mut arena, &[0.25, -0.5]);
        for lb in 0..=10 {
            let runs = h.read_runs(4, lb);
            for (c, k) in runs {
                for col in 0..k {
                    assert_eq!(h.run(&arena, c + col, 1), &[0.25, -0.5]);
                }
            }
        }
    }
}
