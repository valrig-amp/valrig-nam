//! Column-major frame buffers.
//!
//! The reference stores every intermediate as an `Eigen::MatrixXf` of shape
//! `(channels, frames)`. Eigen's default storage order is column-major, so
//! one *frame* is contiguous and one *channel* is strided. Several places in
//! the reference depend on that: `memcpy` of a whole block, `data()` on a
//! `leftCols()` view, activations applied to a flat pointer. So this crate
//! uses the same layout rather than the more Rust-natural row-major one.
//!
//! Buffers are sized once, at [`crate::loader::Model::reset`], and never
//! reallocate during processing.

/// A `(channels, max_frames)` column-major `f32` buffer.
///
/// Only the first `n` columns are meaningful for any given `process` call;
/// the rest is scratch that keeps its previous contents. This mirrors the
/// reference, which also leaves the tail of its buffers stale.
///
/// Every accessor indexes into the backing store and panics when the buffer
/// is smaller than asked; size it with [`Buf::zeros`] or [`Buf::resize`]
/// first.
///
/// ```
/// use valverig_nam::buffer::Buf;
///
/// let mut b = Buf::zeros(2, 4);      // stereo, up to 4 frames
/// b.set(1, 0, 0.5);                  // channel 1, frame 0
/// assert_eq!(b.col(0), &[0.0, 0.5]); // one frame, all channels
/// assert_eq!(b.left(1), &[0.0, 0.5]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Buf {
    rows: usize,
    max_cols: usize,
    data: Vec<f32>,
}

impl Buf {
    /// An empty buffer. Call [`Buf::resize`] before use.
    pub fn new() -> Self {
        Self::default()
    }

    /// A zeroed `(rows, cols)` buffer.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            max_cols: cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Resize and zero. This is the only allocating operation; it belongs in
    /// `reset`, never in `process`.
    pub fn resize(&mut self, rows: usize, cols: usize) {
        self.rows = rows;
        self.max_cols = cols;
        self.data.clear();
        self.data.resize(rows * cols, 0.0);
    }

    /// Number of channels.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Column capacity: the largest `n` any accessor accepts.
    #[inline]
    pub fn cols(&self) -> usize {
        self.max_cols
    }

    /// True when nothing has been allocated.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The whole backing store, column-major.
    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// The whole backing store, mutably.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// The first `n` columns, flat: equivalent to `.leftCols(n).data()`.
    #[inline]
    pub fn left(&self, n: usize) -> &[f32] {
        &self.data[..self.rows * n]
    }

    /// The first `n` columns, flat and mutable.
    #[inline]
    pub fn left_mut(&mut self, n: usize) -> &mut [f32] {
        &mut self.data[..self.rows * n]
    }

    /// One column (one frame, all channels).
    #[inline]
    pub fn col(&self, c: usize) -> &[f32] {
        &self.data[c * self.rows..(c + 1) * self.rows]
    }

    /// One column, mutably.
    #[inline]
    pub fn col_mut(&mut self, c: usize) -> &mut [f32] {
        let r = self.rows;
        &mut self.data[c * r..(c + 1) * r]
    }

    /// Element `(row, col)`.
    #[inline]
    pub fn at(&self, row: usize, col: usize) -> f32 {
        self.data[col * self.rows + row]
    }

    /// Set element `(row, col)`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, v: f32) {
        let r = self.rows;
        self.data[col * r + row] = v;
    }

    /// Zero the first `n` columns.
    #[inline]
    pub(crate) fn zero_left(&mut self, n: usize) {
        self.left_mut(n).fill(0.0);
    }

    /// Fill the first `n` columns from per-channel slices: `input[ch][frame]`.
    ///
    /// Reads exactly [`Buf::rows`] channels and ignores any surplus, which
    /// is what the reference does when a host hands a stereo pair to a mono
    /// capture. Panics when `input` has fewer channels than that or a
    /// channel is shorter than `n`.
    #[inline]
    pub(crate) fn copy_from_channels(&mut self, input: &[&[f32]], n: usize) {
        for (ch, src) in input.iter().take(self.rows).enumerate() {
            for (frame, &v) in src[..n].iter().enumerate() {
                self.set(ch, frame, v);
            }
        }
    }

    /// Write the first `n` columns out to per-channel slices, the inverse of
    /// [`Buf::copy_from_channels`]: exactly [`Buf::rows`] channels, surplus
    /// ignored, a shortfall panics.
    #[inline]
    pub(crate) fn copy_to_channels(&self, output: &mut [&mut [f32]], n: usize) {
        for (ch, dst) in output.iter_mut().take(self.rows).enumerate() {
            for (frame, v) in dst[..n].iter_mut().enumerate() {
                *v = self.at(ch, frame);
            }
        }
    }

    /// `self[..n] += src[..n]`.
    ///
    /// `src` may have more rows than `self`, in which case the *top*
    /// `self.rows` rows of each column are taken, the reference's
    /// `topRows(k)` idiom. This is how a layer array accumulates its layers'
    /// skip outputs straight out of their `z` blocks, gated or not, without
    /// an intermediate copy.
    #[inline]
    pub(crate) fn add_top_from(&mut self, src: &Buf, n: usize) {
        let dr = self.rows;
        let sr = src.rows;
        debug_assert!(dr <= sr);
        for c in 0..n {
            let (d, s) = (
                &mut self.data[c * dr..c * dr + dr],
                &src.data[c * sr..c * sr + dr],
            );
            for (a, b) in d.iter_mut().zip(s) {
                *a += *b;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_major_indexing() {
        let mut b = Buf::zeros(3, 4);
        b.set(0, 0, 1.0);
        b.set(1, 0, 2.0);
        b.set(2, 0, 3.0);
        b.set(0, 1, 4.0);
        assert_eq!(b.data()[..4], [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(b.col(0), &[1.0, 2.0, 3.0]);
        assert_eq!(b.at(0, 1), 4.0);
    }

    #[test]
    fn add_top_takes_leading_rows_of_each_column() {
        let mut src = Buf::zeros(4, 2);
        for c in 0..2 {
            for r in 0..4 {
                src.set(r, c, (c * 4 + r) as f32);
            }
        }
        let mut dst = Buf::zeros(2, 2);
        dst.data_mut().fill(1.0);
        dst.add_top_from(&src, 2);
        // Column 0 rows 0..2 = [0, 1]; column 1 rows 0..2 = [4, 5]; plus the 1.
        assert_eq!(dst.data(), &[1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn channel_slices_round_trip_and_surplus_channels_are_ignored() {
        let mut b = Buf::zeros(2, 3);
        let (l, r, extra) = ([1.0f32, 2.0, 3.0], [4.0f32, 5.0, 6.0], [9.0f32; 3]);
        b.copy_from_channels(&[&l, &r, &extra], 2);
        assert_eq!(b.data(), &[1.0, 4.0, 2.0, 5.0, 0.0, 0.0]);

        let (mut a, mut c, mut d) = ([0.0f32; 3], [0.0f32; 3], [7.0f32; 3]);
        b.copy_to_channels(&mut [&mut a, &mut c, &mut d], 2);
        assert_eq!(a, [1.0, 2.0, 0.0]);
        assert_eq!(c, [4.0, 5.0, 0.0]);
        assert_eq!(d, [7.0; 3], "surplus output channels are left alone");
    }
}
