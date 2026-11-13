//! The LSTM architecture, ported from `NAM/lstm.cpp`.
//!
//! An LSTM model is a stack of cells run one audio sample at a time, followed
//! by an affine head that maps the last layer's hidden state to the output
//! channels. There is no buffering and no receptive field: all of the model's
//! memory lives in the cell and hidden states, which persist across `process`
//! calls.
//!
//! # Layout and association
//!
//! The reference computes the gates as `_ifgo.noalias() = _w * _xh` with
//! `_w` an `Eigen::MatrixXf`. For a column-major matrix Eigen's gemv sweeps
//! columns and accumulates into the result vector, so each output element is
//! `acc = 0; for j { acc += w(i, j) * xh(j) }` with `j` ascending: the plain
//! dot product. `_w` is *filled* row-major ("Assign in row-major because
//! that's how PyTorch goes"); this port transposes it once at load so the
//! gate product can accumulate across the rows as independent sums, one lane
//! each, which keeps every output's addition order and lets the loop run
//! four lanes wide, with the sums in registers from the first column to the
//! last.
//!
//! The bias is a separate pass (`_ifgo += _b`), so it is added after the whole
//! sum, not folded into the accumulator. Same for the head. Nothing here
//! fuses: the gate product is a scalar chain per output where a fused
//! multiply-add is slower, and the reference vectors are pinned unfused.
//!
//! # Weight order
//!
//! Per cell, in this order: the `(4 * hidden, input + hidden)` matrix
//! row-major, the `4 * hidden` bias, the `hidden` **initial hidden state**,
//! then the `hidden` **initial cell state**. The last two are easy to miss -
//! they are model parameters in NAM, not zeros. Then, once for the model, the
//! `(out_channels, hidden)` head matrix row-major and the `out_channels` head
//! bias.

use crate::activations::sigmoid;
use crate::buffer::Buf;
use crate::engine::{self, Engine};
use crate::error::{Error, Result};
use crate::format::LstmConfig;
use crate::lanes::F4;
use crate::weights::WeightReader;

/// Registers of gate sums [`gate_sums`] carries through the columns
/// together: thirty-two gates, with room left for a column of weights.
const GATE_TILE: usize = 8;

/// `R` registers of the gate product, four gates each, from register `at`:
/// `gates = sum_j wt[.., j] * xh[j]` over the columns in order from zero,
/// then the bias.
///
/// `wt` is column-major with `rows` gates a column, and `rows` is a multiple
/// of four. The sums stay in registers across the columns, where
/// accumulating in `gates` itself would send every step through memory and
/// make each wait for the store before it. Each step is a multiply and then
/// an add, two roundings: the reference computes this product unfused and
/// its vectors are pinned that way.
#[inline(always)]
fn gate_sums<const R: usize>(
    wt: &[f32],
    rows: usize,
    xh: &[f32],
    bias: &[[f32; 4]],
    gates: &mut [[f32; 4]],
    at: usize,
) {
    let mut sum = [F4::zero(); R];
    for (col, xj) in wt.chunks_exact(rows).zip(xh) {
        // This column's weights for these gates, checked once here.
        let (w, _) = col[4 * at..4 * (at + R)].as_chunks::<4>();
        let x = F4::splat(*xj);
        for (s, w) in sum.iter_mut().zip(w) {
            *s = s.macc::<false>(F4::load(w), x);
        }
    }
    let bias = &bias[at..at + R];
    for ((g, s), b) in gates[at..at + R].iter_mut().zip(&sum).zip(bias) {
        s.add(F4::load(b)).store(g);
    }
}

/// One LSTM cell: the recurrent unit of `NAM/lstm.cpp`'s `LSTMCell`.
#[derive(Debug, Clone)]
struct LstmCell {
    /// Gate matrix, `(4 * hidden_size)` rows by `(input_size + hidden_size)`
    /// columns, stored column-major.
    wt: Vec<f32>,
    /// Gate bias, `4 * hidden_size`.
    b: Vec<f32>,
    /// Input concatenated with hidden state; the hidden state is the tail.
    xh: Vec<f32>,
    /// Pre-activation gates: input, forget, cell, output at `0, H, 2H, 3H`.
    ifgo: Vec<f32>,
    /// Cell state, `hidden_size`.
    c: Vec<f32>,
    input_size: usize,
    hidden_size: usize,
}

impl LstmCell {
    /// How many floats a cell of this shape consumes from the weight array,
    /// or `None` when that does not fit in `usize`.
    fn weight_count(input_size: usize, hidden_size: usize) -> Option<usize> {
        let rows = hidden_size.checked_mul(4)?;
        let matrix = rows.checked_mul(input_size.checked_add(hidden_size)?)?;
        // Gate bias, initial hidden state, initial cell state.
        matrix
            .checked_add(rows)?
            .checked_add(hidden_size)?
            .checked_add(hidden_size)
    }

    /// Read one cell's parameters from `r`, in the reference's order.
    fn new(input_size: usize, hidden_size: usize, r: &mut WeightReader<'_>) -> Result<Self> {
        let rows = 4 * hidden_size;
        let cols = input_size + hidden_size;

        let mut w = vec![0.0f32; rows * cols];
        // Row-major fill: the outer loop is over rows, matching the reference's
        // `for i in rows { for j in cols { _w(i, j) = *(weights++) } }`.
        r.fill(&mut w)?;
        let mut wt = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                wt[j * rows + i] = w[i * cols + j];
            }
        }
        let mut b = vec![0.0f32; rows];
        r.fill(&mut b)?;
        let mut xh = vec![0.0f32; cols];
        r.fill(&mut xh[input_size..])?;
        let mut c = vec![0.0f32; hidden_size];
        r.fill(&mut c)?;

        Ok(Self {
            wt,
            b,
            xh,
            ifgo: vec![0.0f32; rows],
            c,
            input_size,
            hidden_size,
        })
    }

    /// The current hidden state: the tail of the concatenated `xh` vector.
    fn hidden_state(&self) -> &[f32] {
        &self.xh[self.input_size..]
    }

    /// Advance the cell by one sample. `x` has length `input_size`.
    fn process(&mut self, x: &[f32]) {
        let h = self.hidden_size;
        let cols = self.xh.len();
        debug_assert_eq!(x.len(), self.input_size);

        self.xh[..self.input_size].copy_from_slice(x);

        // Every gate's sum runs over the columns in order from zero, exactly
        // as a row-at-a-time dot product would; only which sums advance
        // together differs, and those are independent. The bias is added to
        // the finished sum, as `_ifgo += _b` does, not to a partial one.
        let rows = 4 * h;
        debug_assert_eq!(self.wt.len(), rows * cols);
        let (gates, _) = self.ifgo.as_chunks_mut::<4>();
        let (bias, _) = self.b.as_chunks::<4>();
        let whole = gates.len() - gates.len() % GATE_TILE;
        let mut at = 0;
        while at < whole {
            gate_sums::<GATE_TILE>(&self.wt, rows, &self.xh, bias, gates, at);
            at += GATE_TILE;
        }
        while at < gates.len() {
            gate_sums::<1>(&self.wt, rows, &self.xh, bias, gates, at);
            at += 1;
        }

        let (i_off, f_off, g_off, o_off) = (0, h, 2 * h, 3 * h);
        // The cell state is updated in place first, and the hidden state is
        // then computed from the *new* cell state, in two separate loops.
        for k in 0..h {
            self.c[k] = sigmoid(self.ifgo[k + f_off]) * self.c[k]
                + sigmoid(self.ifgo[k + i_off]) * self.ifgo[k + g_off].tanh();
        }
        for k in 0..h {
            self.xh[k + self.input_size] = sigmoid(self.ifgo[k + o_off]) * self.c[k].tanh();
        }
    }
}

/// A multi-layer LSTM model with an affine output head.
#[derive(Debug, Clone)]
pub(crate) struct Lstm {
    in_channels: usize,
    out_channels: usize,
    /// `(out_channels, hidden_size)`, row-major.
    head_weight: Vec<f32>,
    head_bias: Vec<f32>,
    layers: Vec<LstmCell>,
    /// Scratch input vector, length `input_size`.
    input: Vec<f32>,
    /// Scratch output vector, length `out_channels`.
    output: Vec<f32>,
    hidden_size: usize,
    expected_sample_rate: Option<f64>,
    max_buffer_size: usize,
    prewarm_on_reset: bool,
}

impl Lstm {
    /// How many floats a model of this shape consumes from the weight array.
    ///
    /// The head is read whatever `num_layers` is, including zero. Fails when
    /// the count does not fit in `usize`.
    fn weight_count(cfg: &LstmConfig) -> Result<usize> {
        let mut n = 0usize;
        for i in 0..cfg.num_layers {
            let input = if i == 0 {
                cfg.input_size
            } else {
                cfg.hidden_size
            };
            n = LstmCell::weight_count(input, cfg.hidden_size)
                .and_then(|c| n.checked_add(c))
                .ok_or_else(|| Error::Config("LSTM has more weights than can be counted".into()))?;
        }
        cfg.out_channels
            .checked_mul(cfg.hidden_size)
            .and_then(|h| h.checked_add(cfg.out_channels))
            .and_then(|h| n.checked_add(h))
            .ok_or_else(|| Error::Config("LSTM has more weights than can be counted".into()))
    }

    /// Build the model and consume `weights` in full.
    ///
    /// `expected_sample_rate` is the file's, in Hz, or `None` when it does
    /// not say; it sets the prewarm length. Fails with
    /// [`Error::WeightCount`] unless `weights` holds exactly as many floats
    /// as the shape needs, checked before anything is allocated, and with
    /// [`Error::Config`] for a zero channel or hidden count, or more input
    /// channels than the cells' input width.
    pub(crate) fn new(
        cfg: &LstmConfig,
        weights: &[f32],
        expected_sample_rate: Option<f64>,
    ) -> Result<Self> {
        // `nam::DSP`'s constructor throws on non-positive channel counts.
        if cfg.in_channels == 0 || cfg.out_channels == 0 {
            return Err(Error::Config("channel counts must be positive".into()));
        }
        if cfg.hidden_size == 0 {
            return Err(Error::Config("LSTM hidden_size must be positive".into()));
        }
        if cfg.in_channels > cfg.input_size {
            // `process` writes `_input(ch)` for `ch < in_channels` into a
            // vector sized `input_size`; the reference would run off the end.
            return Err(Error::Config(format!(
                "LSTM in_channels ({}) exceeds input_size ({})",
                cfg.in_channels, cfg.input_size
            )));
        }
        let needed = Self::weight_count(cfg)?;
        if needed != weights.len() {
            return Err(Error::WeightCount {
                expected: needed,
                found: weights.len(),
            });
        }

        let mut r = WeightReader::new(weights);
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let cell_input = if i == 0 {
                cfg.input_size
            } else {
                cfg.hidden_size
            };
            layers.push(LstmCell::new(cell_input, cfg.hidden_size, &mut r)?);
        }

        let mut head_weight = vec![0.0f32; cfg.out_channels * cfg.hidden_size];
        r.fill(&mut head_weight)?;
        let mut head_bias = vec![0.0f32; cfg.out_channels];
        r.fill(&mut head_bias)?;
        r.finish()?;

        Ok(Self {
            in_channels: cfg.in_channels,
            out_channels: cfg.out_channels,
            head_weight,
            head_bias,
            layers,
            // The reference leaves `_input` uninitialised past `in_channels`.
            // Zeroing is the only defined choice; a model with
            // `input_size > in_channels` is not reproducible either way.
            input: vec![0.0f32; cfg.input_size],
            output: vec![0.0f32; cfg.out_channels],
            hidden_size: cfg.hidden_size,
            expected_sample_rate,
            max_buffer_size: 0,
            // `nam::DSP`'s constructor takes `gPrewarmOnResetDefault`, true
            // unless a host has scoped it off.
            prewarm_on_reset: true,
        })
    }

    /// One sample through the stack and the head, leaving the result in
    /// `output`.
    fn process_sample(&mut self) {
        if self.layers.is_empty() {
            // Degenerate model: pass through what channels line up, zero the rest.
            let n = self.in_channels.min(self.out_channels);
            self.output[..n].copy_from_slice(&self.input[..n]);
            for v in &mut self.output[n..] {
                *v = 0.0;
            }
            return;
        }

        self.layers[0].process(&self.input);
        for i in 1..self.layers.len() {
            // Split so the previous layer's hidden state can be borrowed while
            // this one is advanced; the reference passes an Eigen::Ref for the
            // same reason: no copy of the hidden state.
            let (earlier, rest) = self.layers.split_at_mut(i);
            rest[0].process(earlier[i - 1].hidden_state());
        }

        let hidden = self.layers[self.layers.len() - 1].hidden_state();
        let h = self.hidden_size;
        for (oc, out) in self.output.iter_mut().enumerate() {
            let row = &self.head_weight[oc * h..oc * h + h];
            let mut acc = 0.0f32;
            for (wk, hk) in row.iter().zip(hidden.iter()) {
                acc += wk * hk;
            }
            *out = acc;
        }
        for (v, bias) in self.output.iter_mut().zip(self.head_bias.iter()) {
            *v += *bias;
        }
    }
}

impl Engine for Lstm {
    fn in_channels(&self) -> usize {
        self.in_channels
    }

    fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Half a second at the file's sample rate, *"Hacky, but a half-second
    /// seems to work for most models."*, and 1 when the file states no rate,
    /// so that *something* happens.
    fn prewarm_samples(&self) -> usize {
        match self.expected_sample_rate {
            Some(sr) => ((0.5 * sr) as usize).max(1),
            None => 1,
        }
    }

    fn max_buffer_size(&self) -> usize {
        self.max_buffer_size
    }

    /// Record the largest block `process` will be called with.
    ///
    /// An LSTM has nothing to size, since it runs one sample at a time, so
    /// this only stores the value, as `nam::DSP::SetMaxBufferSize` does. It still
    /// matters: `nam::DSP::prewarm` pushes silence in blocks of this size, so
    /// it rounds the prewarm length up to a multiple of it.
    fn set_max_buffer_size(&mut self, max_buffer: usize) {
        self.max_buffer_size = max_buffer;
    }

    fn set_prewarm_on_reset(&mut self, on: bool) {
        self.prewarm_on_reset = on;
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    /// Settle the model on silence: `nam::DSP::prewarm`, unmodified.
    ///
    /// Allocating; never call it from the audio thread. The block-sized
    /// overshoot in [`engine::prewarm_with_silence`] is not hypothetical for
    /// a recurrent model: those extra samples land in the hidden state.
    fn prewarm(&mut self) {
        let max_buffer = self.max_buffer_size;
        engine::prewarm_with_silence(self, max_buffer);
    }

    /// Process `num_frames` frames, sample by sample.
    ///
    /// The reference's buffers are `double`; narrowing to `f32` is the
    /// caller's job here, and is what `_input(ch) = input[ch][i]` does there.
    fn process(&mut self, input: &[&[f32]], output: &mut [&mut [f32]], num_frames: usize) {
        assert!(input.len() >= self.in_channels, "not enough input channels");
        assert!(
            output.len() >= self.out_channels,
            "not enough output channels"
        );
        for i in 0..num_frames {
            // `self.input` may be longer than `in_channels`; zip stops at the
            // shorter, which is the reference's `ch < in_channels` bound.
            for (dst, src) in self.input.iter_mut().zip(input[..self.in_channels].iter()) {
                *dst = src[i];
            }
            self.process_sample();
            for (src, dst) in self.output.iter().zip(output.iter_mut()) {
                dst[i] = *src;
            }
        }
    }

    fn process_buf(&mut self, input: &Buf, output: &mut Buf, num_frames: usize) {
        debug_assert_eq!(input.rows(), self.in_channels);
        debug_assert_eq!(output.rows(), self.out_channels);
        for i in 0..num_frames {
            // Bounded by `in_channels` exactly as `process` is.
            let col = &input.col(i)[..self.in_channels];
            for (dst, src) in self.input.iter_mut().zip(col.iter()) {
                *dst = *src;
            }
            self.process_sample();
            let out = output.col_mut(i);
            for (dst, src) in out.iter_mut().zip(self.output.iter()) {
                *dst = *src;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::splitmix_stream;

    fn config(
        num_layers: usize,
        input_size: usize,
        hidden_size: usize,
        out_channels: usize,
    ) -> LstmConfig {
        LstmConfig {
            num_layers,
            input_size,
            hidden_size,
            in_channels: 1,
            out_channels,
        }
    }

    fn lstm(cfg: &LstmConfig, w: &[f32]) -> Result<Lstm> {
        Lstm::new(cfg, w, Some(48_000.0))
    }

    fn run(model: &mut Lstm, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        model.process(&[x], &mut [&mut out], x.len());
        out
    }

    // -----------------------------------------------------------------------
    // An independent transcription of the LSTM equations, written from the
    // definition rather than from the code above: a different data layout
    // (one Vec per row), a different traversal, and gates named explicitly.
    // -----------------------------------------------------------------------

    struct DirectCell {
        rows: Vec<Vec<f32>>, // 4H rows of (input + H)
        bias: Vec<f32>,
        h: Vec<f32>,
        c: Vec<f32>,
    }

    impl DirectCell {
        fn read(input_size: usize, hidden: usize, w: &[f32], at: &mut usize) -> Self {
            let cols = input_size + hidden;
            let mut rows = Vec::new();
            for _ in 0..4 * hidden {
                let row = w[*at..*at + cols].to_vec();
                *at += cols;
                rows.push(row);
            }
            let bias = w[*at..*at + 4 * hidden].to_vec();
            *at += 4 * hidden;
            let h = w[*at..*at + hidden].to_vec();
            *at += hidden;
            let c = w[*at..*at + hidden].to_vec();
            *at += hidden;
            Self { rows, bias, h, c }
        }

        fn step(&mut self, x: &[f32]) {
            let hidden = self.h.len();
            let mut xh = x.to_vec();
            xh.extend_from_slice(&self.h);
            let gates: Vec<f32> = self
                .rows
                .iter()
                .zip(self.bias.iter())
                .map(|(row, b)| {
                    let mut acc = 0.0f32;
                    for (wj, xj) in row.iter().zip(xh.iter()) {
                        acc += wj * xj;
                    }
                    acc + b
                })
                .collect();
            let sig = |v: f32| 1.0f32 / (1.0f32 + (-v).exp());
            for k in 0..hidden {
                let i_gate = sig(gates[k]);
                let f_gate = sig(gates[hidden + k]);
                let g_gate = gates[2 * hidden + k].tanh();
                self.c[k] = f_gate * self.c[k] + i_gate * g_gate;
            }
            for k in 0..hidden {
                let o_gate = sig(gates[3 * hidden + k]);
                self.h[k] = o_gate * self.c[k].tanh();
            }
        }
    }

    fn direct_forward(cfg: &LstmConfig, w: &[f32], xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let hidden = cfg.hidden_size;
        let mut at = 0usize;
        let mut cells: Vec<DirectCell> = (0..cfg.num_layers)
            .map(|i| {
                DirectCell::read(
                    if i == 0 { cfg.input_size } else { hidden },
                    hidden,
                    w,
                    &mut at,
                )
            })
            .collect();
        let head: Vec<Vec<f32>> = (0..cfg.out_channels)
            .map(|_| {
                let row = w[at..at + hidden].to_vec();
                at += hidden;
                row
            })
            .collect();
        let head_bias = w[at..at + cfg.out_channels].to_vec();
        at += cfg.out_channels;
        assert_eq!(at, w.len(), "direct model must consume every weight");

        let mut out = Vec::new();
        for x in xs {
            let mut v = x.clone();
            for cell in cells.iter_mut() {
                cell.step(&v);
                v = cell.h.clone();
            }
            let frame = head
                .iter()
                .zip(head_bias.iter())
                .map(|(row, b)| {
                    let mut acc = 0.0f32;
                    for (wk, hk) in row.iter().zip(v.iter()) {
                        acc += wk * hk;
                    }
                    acc + b
                })
                .collect();
            out.push(frame);
        }
        out
    }

    #[test]
    fn matches_direct_definition() {
        // Two layers, two output channels: exercises the layer hop and the head.
        let cfg = config(2, 1, 4, 2);
        let n = Lstm::weight_count(&cfg).unwrap();
        let stream = splitmix_stream(0x5EED_1234, n + 97);
        let w: Vec<f32> = stream[..n].iter().map(|u| 0.5 * u).collect();
        let xs: Vec<Vec<f32>> = stream[n..].iter().map(|u| vec![0.4 * u]).collect();
        let expected = direct_forward(&cfg, &w, &xs);

        let mut model = lstm(&cfg, &w).unwrap();
        let mut got = vec![vec![0.0f32; xs.len()]; cfg.out_channels];
        {
            let in_ch: Vec<f32> = xs.iter().map(|v| v[0]).collect();
            let input: Vec<&[f32]> = vec![&in_ch];
            let mut refs: Vec<&mut [f32]> = got.iter_mut().map(|c| c.as_mut_slice()).collect();
            model.process(&input, &mut refs, xs.len());
        }

        for (i, frame) in expected.iter().enumerate() {
            for (ch, want) in frame.iter().enumerate() {
                assert_eq!(
                    got[ch][i].to_bits(),
                    want.to_bits(),
                    "frame {i} channel {ch}: {} vs {want}",
                    got[ch][i]
                );
            }
        }
    }

    #[test]
    fn ragged_blocks_match_one_at_a_time() {
        // State must carry across calls, so any block schedule gives the same
        // samples as feeding them one by one.
        let cfg = config(1, 1, 3, 1);
        let n = Lstm::weight_count(&cfg).unwrap();
        let stream = splitmix_stream(99, n + 200);
        let w: Vec<f32> = stream[..n].iter().map(|u| 0.5 * u).collect();
        let x: Vec<f32> = stream[n..].iter().map(|u| 0.3 * u).collect();

        let whole = run(&mut lstm(&cfg, &w).unwrap(), &x);

        let mut b = lstm(&cfg, &w).unwrap();
        let mut piecewise = vec![0.0f32; x.len()];
        let mut pos = 0usize;
        for size in [1usize, 7, 64, 3, 128, 17].iter().cycle() {
            if pos >= x.len() {
                break;
            }
            let n = (*size).min(x.len() - pos);
            b.process(&[&x[pos..pos + n]], &mut [&mut piecewise[pos..pos + n]], n);
            pos += n;
        }

        assert_eq!(whole, piecewise);
    }

    #[test]
    fn gate_order_and_row_major_layout() {
        // hidden = 1, input = 1, so the matrix is 4 rows of [w_x, w_h] and each
        // row is one gate. Distinct values make any transposition or reordering
        // of i/f/g/o visible.
        let (wx_i, wh_i) = (0.7f32, 0.11f32);
        let (wx_f, wh_f) = (-0.3f32, 0.23f32);
        let (wx_g, wh_g) = (1.3f32, -0.37f32);
        let (wx_o, wh_o) = (0.9f32, 0.41f32);
        let (h0, c0) = (0.25f32, -0.6f32);
        let mut w = vec![
            wx_i, wh_i, // row 0: input gate
            wx_f, wh_f, // row 1: forget gate
            wx_g, wh_g, // row 2: cell ("g") gate
            wx_o, wh_o, // row 3: output gate
        ];
        let bias = [0.05f32, -0.15, 0.35, -0.45];
        w.extend_from_slice(&bias);
        w.push(h0);
        w.push(c0);
        w.push(1.0); // head weight: read the hidden state straight out
        w.push(0.0); // head bias

        let mut model = lstm(&config(1, 1, 1, 1), &w).unwrap();
        // `black_box`: with a constant input LLVM folds the `expf`/`tanhf` in
        // the expected value below through its double-precision folder, which
        // can land one ULP away from the runtime libm the model calls.
        let x = std::hint::black_box(0.8f32);
        let out = run(&mut model, &[x]);

        let sig = |v: f32| 1.0f32 / (1.0f32 + (-v).exp());
        let i_gate = sig(wx_i * x + wh_i * h0 + bias[0]);
        let f_gate = sig(wx_f * x + wh_f * h0 + bias[1]);
        let g_gate = (wx_g * x + wh_g * h0 + bias[2]).tanh();
        let o_gate = sig(wx_o * x + wh_o * h0 + bias[3]);
        let c1 = f_gate * c0 + i_gate * g_gate;
        let h1 = o_gate * c1.tanh();

        assert_eq!(out[0].to_bits(), h1.to_bits(), "{} vs {h1}", out[0]);
    }

    #[test]
    fn initial_states_come_from_the_weights() {
        // Every gate weight is zero except the output gate's hidden column, so
        // the first sample's result is sigmoid(h0) * tanh(0.5 * c0). Dropping
        // h0 would give sigmoid(0) instead; dropping c0 would give 0.
        // `black_box` for the same reason as in `gate_order_and_row_major_layout`.
        let (h0, c0) = std::hint::black_box((0.75f32, -0.5f32));
        let w = [
            0.0, 0.0, // input gate
            0.0, 0.0, // forget gate
            0.0, 0.0, // cell gate
            0.0, 1.0, // output gate: reads the hidden state
            0.0, 0.0, 0.0, 0.0, // bias
            h0, c0,  // initial hidden state, initial cell state
            1.0, // head weight
            0.0, // head bias
        ];

        let mut model = lstm(&config(1, 1, 1, 1), &w).unwrap();
        let out = run(&mut model, &[0.0]);

        let sig = |v: f32| 1.0f32 / (1.0f32 + (-v).exp());
        let c1 = sig(0.0) * c0 + sig(0.0) * 0.0f32.tanh();
        let h1 = sig(h0) * c1.tanh();
        assert_eq!(out[0].to_bits(), h1.to_bits(), "{} vs {h1}", out[0]);
    }

    #[test]
    fn zero_layers_pass_input_through() {
        // Still reads a head: (out_channels x hidden) + out_channels.
        let hidden = 3usize;
        let w = vec![0.0f32; 2 * hidden + 2];
        let mut model = lstm(&config(0, 1, hidden, 2), &w).unwrap();
        let xs = [0.125f32, -0.25, 0.5];
        let mut a = vec![0.0f32; 3];
        let mut b = vec![9.0f32; 3];
        model.process(&[&xs], &mut [&mut a, &mut b], 3);
        assert_eq!(a, xs.to_vec(), "channel 0 is copied");
        assert_eq!(b, vec![0.0f32; 3], "surplus output channels are zeroed");
    }

    #[test]
    fn prewarm_samples_follows_the_reference() {
        let cfg = config(1, 1, 1, 1);
        let w = vec![0.0f32; Lstm::weight_count(&cfg).unwrap()];
        let at = |sr: Option<f64>| Lstm::new(&cfg, &w, sr).unwrap().prewarm_samples();
        assert_eq!(at(Some(48000.0)), 24000);
        assert_eq!(at(Some(44100.0)), 22050);
        assert_eq!(at(Some(1.0)), 1, "0.5 truncates to 0, then clamps up to 1");
        assert_eq!(at(None), 1, "unknown sample rate");
    }

    #[test]
    fn weight_count_is_enforced_in_both_directions() {
        let cfg = config(1, 1, 3, 1);
        let n = Lstm::weight_count(&cfg).unwrap();
        assert_eq!(n, 70, "matches assets/models/lstm.nam");
        let w = vec![0.0f32; n];
        assert!(lstm(&cfg, &w).is_ok());
        assert!(matches!(
            lstm(&cfg, &w[..n - 1]),
            Err(Error::WeightCount {
                expected: 70,
                found: 69
            })
        ));
        let mut too_many = w.clone();
        too_many.push(0.0);
        assert!(matches!(
            lstm(&cfg, &too_many),
            Err(Error::WeightCount {
                expected: 70,
                found: 71
            })
        ));
    }

    #[test]
    fn rejects_channel_counts_the_reference_would_break_on() {
        let w = vec![0.0f32; Lstm::weight_count(&config(1, 1, 1, 1)).unwrap()];
        let with = |in_channels, out_channels| LstmConfig {
            num_layers: 1,
            input_size: 1,
            hidden_size: 1,
            in_channels,
            out_channels,
        };
        assert!(matches!(lstm(&with(0, 1), &w), Err(Error::Config(_))));
        assert!(matches!(lstm(&with(1, 0), &w), Err(Error::Config(_))));
        // in_channels > input_size would write past the end of `_input`.
        assert!(matches!(lstm(&with(2, 1), &w), Err(Error::Config(_))));
        // A hidden size of zero has no gates to run.
        assert!(matches!(
            lstm(&config(1, 1, 0, 1), &[0.0, 0.0][..0]),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn an_absurd_shape_is_refused_before_anything_is_allocated() {
        // 2^20 hidden units: the gate matrix alone is 4 T floats. The count
        // is compared against the file before a single vector is sized.
        let cfg = config(1, 1, 1 << 20, 1);
        assert!(matches!(
            lstm(&cfg, &[0.0; 4]),
            Err(Error::WeightCount { found: 4, .. })
        ));
    }
}
