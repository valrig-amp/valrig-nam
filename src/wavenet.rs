//! The WaveNet architecture: the model behind essentially every modern
//! `.nam` capture.
//!
//! Ported from `NAM/wavenet/model.cpp`, `model.h`, `detail.h` and `params.h`.
//! The wiring, the buffer shapes, the order of floating-point operations and
//! the order in which weights are drawn from the file all follow the
//! reference exactly; see `README.md` for what "exactly" is measured
//! against.
//!
//! A model is a stack of *layer arrays*. Each array projects its input to
//! `channels`, runs a chain of dilated-convolution layers with residual
//! connections, accumulates every layer's skip output, and projects that sum
//! to `head_size` for the next array. The last array's head output is scaled
//! by `head_scale` and, optionally, run through a post-stack head.
//!
//! A layer is built straight from the file's [`LayerArrayConfig`] and its
//! [`LayerConfig`] entry, so there is one description of a layer array, not a
//! parsed one and a runtime one kept in step by hand; [`film_site`] indexes
//! the FiLM array on both sides.
//!
//! [`Layer`] runs any layer the format can describe. The layers of an array
//! that uses none of the options, which is every layer of an A1 or A2
//! capture, are then handed to [`crate::fused`], which computes the same
//! numbers in a quarter of the passes; [`WaveNet::new`] builds the model
//! without that step, and [`WaveNet::fused`] takes it.

use crate::activations::Activation;
use crate::buffer::Buf;
use crate::conv::{Conv1D, Conv1x1, View};
use crate::engine::{self, Engine};
use crate::error::{Error, Result};
use crate::film::Film;
use crate::format::{
    FilmConfig, GatingMode, LayerArrayConfig, LayerConfig, MAX_HISTORY_FLOATS, PostStackHeadConfig,
    WaveNetConfig, film_site,
};
use crate::fused::{FusedLayers, PlainLayer};
use crate::gating::Nonlinearity;
use crate::history::Arena;
use crate::weights::WeightReader;

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Layer {
    conv: Conv1D,
    input_mixin: Conv1x1,
    layer1x1: Option<Conv1x1>,
    head1x1: Option<Conv1x1>,
    z: Buf,
    output_next_layer: Buf,
    nonlinearity: Nonlinearity,
    bottleneck: usize,
    films: [Option<Film>; 8],
}

/// The `Layer` constructor's own checks, which need only the config.
fn check_layer_array(a: &LayerArrayConfig) -> Result<()> {
    if !a.layer1x1.active {
        if a.bottleneck != a.channels {
            return Err(Error::Config(format!(
                "When layer1x1.active is false, bottleneck ({}) must equal channels ({})",
                a.bottleneck, a.channels
            )));
        }
        if a.films[film_site::LAYER1X1_POST].active {
            return Err(Error::Config(
                "layer1x1_post_film cannot be active when layer1x1 is not active".into(),
            ));
        }
    }
    if !a.head1x1.active && a.films[film_site::HEAD1X1_POST].active {
        return Err(Error::Config(
            "Do not use post-head 1x1 FiLM if there is no head 1x1".into(),
        ));
    }
    if a.layers.is_empty() {
        return Err(Error::Config("layer array has no layers".into()));
    }
    Ok(())
}

impl Layer {
    /// Build one layer of array `a` from its per-layer entry `l`, reading its
    /// weights from `r` in the reference's order: the dilated convolution,
    /// the input mixin, the residual 1x1, the head 1x1, then each active
    /// FiLM site.
    fn new(a: &LayerArrayConfig, l: &LayerConfig, r: &mut WeightReader<'_>) -> Result<Self> {
        let gated = l.gating_mode != GatingMode::None;
        let z_channels = if gated {
            2 * a.bottleneck
        } else {
            a.bottleneck
        };

        let conv = Conv1D::new(
            a.channels,
            z_channels,
            l.kernel_size,
            true,
            l.dilation,
            a.groups_input,
            r,
        )?;
        let input_mixin =
            Conv1x1::new(a.condition_size, z_channels, false, a.groups_input_mixin, r)?;
        let layer1x1 = if a.layer1x1.active {
            Some(Conv1x1::new(
                a.bottleneck,
                a.channels,
                true,
                a.layer1x1.groups,
                r,
            )?)
        } else {
            None
        };
        let head1x1 = if a.head1x1.active {
            Some(Conv1x1::new(
                a.bottleneck,
                a.head1x1.out_channels,
                true,
                a.head1x1.groups,
                r,
            )?)
        } else {
            None
        };

        // Each FiLM site modulates a different width; the reference picks
        // them in the Layer constructor and they are easy to get subtly
        // wrong. The two post-1x1 sites are only reachable with their 1x1
        // present, which `check_layer_array` guarantees.
        let widths = [
            a.channels,
            z_channels,
            a.condition_size,
            z_channels,
            z_channels,
            a.bottleneck,
            a.channels,
            a.head1x1.out_channels,
        ];
        let mut films: [Option<Film>; 8] = Default::default();
        for (site, width) in widths.into_iter().enumerate() {
            let fp: &FilmConfig = &a.films[site];
            if fp.active {
                films[site] = Some(Film::new(a.condition_size, width, fp.shift, fp.groups, r)?);
            }
        }

        Ok(Self {
            conv,
            input_mixin,
            layer1x1,
            head1x1,
            z: Buf::new(),
            output_next_layer: Buf::new(),
            nonlinearity: Nonlinearity::new(
                l.gating_mode,
                l.activation.clone(),
                l.secondary_activation.clone(),
                a.bottleneck,
            ),
            bottleneck: a.bottleneck,
            films,
        })
    }

    /// The layer's parts if it is plain at `c` channels: one ungated
    /// activation, a plain convolution and a plain 1x1 of that width with
    /// their biases, a biasless mixin from one condition channel, and no
    /// bottleneck, head 1x1 or FiLM. Anything else is `None`.
    fn plain_parts(&self, c: usize) -> Option<PlainLayer> {
        let Nonlinearity::Plain(activation) = &self.nonlinearity else {
            return None;
        };
        let conv = &self.conv;
        let mixin = &self.input_mixin;
        let l1 = self.layer1x1.as_ref()?;
        if self.head1x1.is_some()
            || self.films.iter().any(Option::is_some)
            || self.bottleneck != c
            || conv.in_channels() != c
            || conv.out_channels() != c
            || !conv.is_plain()
            || conv.bias().len() != c
            || conv.kernel_size() == 0
            || mixin.in_channels() != 1
            || mixin.out_channels() != c
            || !mixin.is_plain()
            || !mixin.bias().is_empty()
            || l1.in_channels() != c
            || l1.out_channels() != c
            || !l1.is_plain()
            || l1.bias().len() != c
        {
            return None;
        }
        Some(PlainLayer {
            kernel_size: conv.kernel_size(),
            dilation: conv.dilation(),
            conv_w: conv.tap_matrices(),
            conv_b: conv.bias().to_vec(),
            mixin_w: mixin.weight().to_vec(),
            l1x1_w: l1.weight().to_vec(),
            l1x1_b: l1.bias().to_vec(),
            activation: activation.clone(),
        })
    }

    /// What the layer contributes to the skip path: the `head1x1` output when
    /// there is one, otherwise `z` itself, whose top `bottleneck` rows the
    /// caller accumulates. Handed on by reference; nothing is copied.
    fn head_output(&self) -> &Buf {
        match &self.head1x1 {
            Some(h) => &h.output,
            None => &self.z,
        }
    }

    fn set_max_buffer_size(&mut self, arena: &mut Arena, n: usize) {
        self.conv.set_max_buffer_size(arena, n);
        self.input_mixin.set_max_buffer_size(n);
        self.z.resize(self.conv.out_channels(), n);
        if let Some(c) = &mut self.layer1x1 {
            c.set_max_buffer_size(n);
        }
        self.output_next_layer.resize(self.conv.in_channels(), n);
        if let Some(h) = &mut self.head1x1 {
            h.set_max_buffer_size(n);
        }
        for f in self.films.iter_mut().flatten() {
            f.set_max_buffer_size(n);
        }
    }

    fn process(&mut self, arena: &mut Arena, input: &Buf, condition: &Buf, n: usize) {
        use film_site::*;

        // Step 1: the dilated convolution and the condition mixin.
        if let Some(f) = self.films[CONV_PRE].as_mut() {
            f.process(View::full(input), condition, n);
            self.conv.process(arena, &f.output, n);
        } else {
            self.conv.process(arena, input, n);
        }
        if let Some(f) = self.films[CONV_POST].as_mut() {
            f.process_in_place(&mut self.conv.output, condition, n);
        }

        if let Some(f) = self.films[INPUT_MIXIN_PRE].as_mut() {
            f.process(View::full(condition), condition, n);
            self.input_mixin.process(&f.output, n);
        } else {
            self.input_mixin.process(condition, n);
        }
        if let Some(f) = self.films[INPUT_MIXIN_POST].as_mut() {
            f.process_in_place(&mut self.input_mixin.output, condition, n);
        }

        {
            let z = self.z.left_mut(n);
            let a = self.conv.output.left(n);
            let b = self.input_mixin.output.left(n);
            for i in 0..z.len() {
                z[i] = a[i] + b[i];
            }
        }

        if let Some(f) = self.films[ACTIVATION_PRE].as_mut() {
            f.process_in_place(&mut self.z, condition, n);
        }

        // Step 2 and 3: activation, then the two 1x1 convolutions. From here
        // on only the top `bottleneck` rows of `z` carry signal; when the
        // layer is not gated that is all of `z`, so one path serves both.
        let bn = self.bottleneck;
        self.nonlinearity.apply(&mut self.z, n);
        if let Some(f) = self.films[ACTIVATION_POST].as_mut() {
            // The reference runs the non-in-place form on a `topRows` block
            // and copies the result back over those rows.
            f.process(View::top_rows(&self.z, bn), condition, n);
            let src = f.output.left(n);
            let rows = self.z.rows();
            for c in 0..n {
                self.z.data_mut()[c * rows..c * rows + bn]
                    .copy_from_slice(&src[c * bn..(c + 1) * bn]);
            }
        }
        if let Some(c) = &mut self.layer1x1 {
            c.process_view(View::top_rows(&self.z, bn), n);
            if let Some(f) = self.films[LAYER1X1_POST].as_mut() {
                f.process_in_place(&mut c.output, condition, n);
            }
        }

        // Skip connection. `head_output()` hands the result on by reference.
        if let Some(h) = &mut self.head1x1 {
            h.process_view(View::top_rows(&self.z, bn), n);
            if let Some(f) = self.films[HEAD1X1_POST].as_mut() {
                f.process_in_place(&mut h.output, condition, n);
            }
        }

        // Residual connection.
        let out = self.output_next_layer.left_mut(n);
        match &self.layer1x1 {
            Some(c) => {
                let a = input.left(n);
                let b = c.output.left(n);
                for i in 0..out.len() {
                    out[i] = a[i] + b[i];
                }
            }
            None => out.copy_from_slice(input.left(n)),
        }
    }
}

// ---------------------------------------------------------------------------
// LayerArray
// ---------------------------------------------------------------------------

/// The chain of layers in an array, in one of the two forms that run it.
#[derive(Debug, Clone)]
enum Layers {
    /// Any layers the format can describe, each run on its own: what
    /// [`LayerArray::new`] builds.
    Each(Vec<Layer>),
    /// Plain layers run together; see [`crate::fused`].
    Fused(FusedLayers),
}

impl Layers {
    /// The convolutions of layers that run on their own, in order; fused
    /// layers have none to show.
    fn convs(&self) -> impl Iterator<Item = &Conv1D> {
        let each: &[Layer] = match self {
            Layers::Each(layers) => layers,
            Layers::Fused(_) => &[],
        };
        each.iter().map(|l| &l.conv)
    }

    fn set_max_buffer_size(&mut self, arena: &mut Arena, n: usize) {
        match self {
            Layers::Each(layers) => {
                for l in layers {
                    l.set_max_buffer_size(arena, n);
                }
            }
            Layers::Fused(f) => f.set_max_buffer_size(n),
        }
    }

    /// Run the chain over `n` frames of `input`, adding every layer's skip
    /// output to `skips`.
    fn process(
        &mut self,
        arena: &mut Arena,
        input: &Buf,
        condition: &Buf,
        skips: &mut Buf,
        n: usize,
    ) {
        match self {
            Layers::Each(layers) => {
                let mut input = input;
                for layer in layers {
                    layer.process(arena, input, condition, n);
                    let layer: &Layer = layer;
                    skips.add_top_from(layer.head_output(), n);
                    input = &layer.output_next_layer;
                }
            }
            Layers::Fused(f) => f.process(input, condition, skips, n),
        }
    }

    /// The last layer's output.
    fn output(&self) -> &Buf {
        match self {
            Layers::Each(layers) => {
                &layers
                    .last()
                    .expect("layer array has at least one layer")
                    .output_next_layer
            }
            Layers::Fused(f) => f.output(),
        }
    }

    fn has_cached_prewarm_state(&self) -> bool {
        match self {
            Layers::Each(_) => self.convs().all(Conv1D::has_cached_prewarm_state),
            Layers::Fused(f) => f.has_cached_prewarm_state(),
        }
    }

    fn prewarm_from_cache(&mut self, arena: &mut Arena) {
        match self {
            Layers::Each(layers) => {
                for l in layers {
                    l.conv.prewarm_from_cache(arena);
                }
            }
            Layers::Fused(f) => f.prewarm_from_cache(),
        }
    }

    fn cache_prewarm_state(&mut self, arena: &Arena) {
        match self {
            Layers::Each(layers) => {
                for l in layers {
                    l.conv.cache_prewarm_state(arena);
                }
            }
            Layers::Fused(f) => f.cache_prewarm_state(),
        }
    }
}

#[derive(Debug, Clone)]
struct LayerArray {
    rechannel: Conv1x1,
    layers: Layers,
    head_inputs: Buf,
    head_rechannel: Conv1D,
    head_output_size: usize,
}

/// Skip channels a layer array accumulates: the width of each layer's head
/// output, which the optional `head1x1` sets when it is active.
fn head_output_size(p: &LayerArrayConfig) -> usize {
    if p.head1x1.active {
        p.head1x1.out_channels
    } else {
        p.bottleneck
    }
}

impl LayerArray {
    /// Build from its config, reading weights in the reference's order: the
    /// rechannel, every layer, the head rechannel.
    fn new(p: &LayerArrayConfig, r: &mut WeightReader<'_>) -> Result<Self> {
        let head_output_size = head_output_size(p);
        let rechannel = Conv1x1::new(p.input_size, p.channels, false, 1, r)?;
        let layers = p
            .layers
            .iter()
            .map(|l| Layer::new(p, l, r))
            .collect::<Result<Vec<_>>>()?;
        let head_rechannel = Conv1D::new(
            head_output_size,
            p.head_size,
            p.head_kernel_size,
            p.head_bias,
            p.head_dilation,
            1,
            r,
        )?;
        Ok(Self {
            rechannel,
            layers: Layers::Each(layers),
            head_inputs: Buf::new(),
            head_rechannel,
            head_output_size,
        })
    }

    /// Put the layers in their fused form if every one of them is plain;
    /// otherwise leave the array as it is. `wants_output` says whether
    /// another array follows and reads this one's last layer.
    ///
    /// For a model that has not been sized yet: the fused layers start with
    /// no history, as the layers they replace do.
    fn fuse(&mut self, wants_output: bool) {
        let Layers::Each(layers) = &self.layers else {
            return;
        };
        let c = self.rechannel.out_channels();
        let parts: Option<Vec<PlainLayer>> = layers.iter().map(|l| l.plain_parts(c)).collect();
        if let Some(fused) = parts.and_then(|p| FusedLayers::new(c, p, wants_output)) {
            self.layers = Layers::Fused(fused);
        }
    }

    /// Every convolution the array runs through `Conv1D`, in execution
    /// order: the layers', unless they have been fused, and the head's.
    fn convs(&self) -> impl Iterator<Item = &Conv1D> {
        self.layers
            .convs()
            .chain(std::iter::once(&self.head_rechannel))
    }

    /// The reference's `get_receptive_field`: how far back the array reaches.
    fn receptive_field(&self) -> Result<usize> {
        self.convs().try_fold(
            0usize,
            |acc, c| Ok(acc.saturating_add(c.receptive_field()?)),
        )
    }

    fn set_max_buffer_size(&mut self, arena: &mut Arena, n: usize) {
        self.rechannel.set_max_buffer_size(n);
        self.layers.set_max_buffer_size(arena, n);
        // Sized after the layers, so that a head history in the arena lands
        // next to the layer histories, in execution order.
        self.head_rechannel.set_max_buffer_size(arena, n);
        self.head_inputs.resize(self.head_output_size, n);
    }

    fn head_outputs(&self) -> &Buf {
        &self.head_rechannel.output
    }

    /// Run the array over `n` frames.
    ///
    /// The skip accumulator starts from the previous array's head output, or
    /// from zero for the first array, following `_process_first` and
    /// `Process` upstream, which differ only in that seeding.
    fn process(
        &mut self,
        arena: &mut Arena,
        inputs: &Buf,
        condition: &Buf,
        head_inputs: Option<&Buf>,
        n: usize,
    ) {
        match head_inputs {
            Some(h) => self.head_inputs.left_mut(n).copy_from_slice(h.left(n)),
            None => self.head_inputs.zero_left(n),
        }
        self.rechannel.process(inputs, n);
        self.layers.process(
            arena,
            &self.rechannel.output,
            condition,
            &mut self.head_inputs,
            n,
        );
        self.head_rechannel.process(arena, &self.head_inputs, n);
    }

    /// The last layer's output, which the next array takes as its input.
    fn layer_outputs(&self) -> &Buf {
        self.layers.output()
    }

    fn has_cached_prewarm_state(&self) -> bool {
        self.layers.has_cached_prewarm_state() && self.head_rechannel.has_cached_prewarm_state()
    }

    fn prewarm_from_cache(&mut self, arena: &mut Arena) {
        self.layers.prewarm_from_cache(arena);
        self.head_rechannel.prewarm_from_cache(arena);
    }

    fn cache_state_as_prewarmed(&mut self, arena: &Arena) {
        self.layers.cache_prewarm_state(arena);
        self.head_rechannel.cache_prewarm_state(arena);
    }
}

// ---------------------------------------------------------------------------
// Post-stack head
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Head {
    convs: Vec<Conv1D>,
    activation: Activation,
    in_channels: usize,
}

/// The `Head` constructor's own checks, which need only the config.
fn check_head(p: &PostStackHeadConfig) -> Result<()> {
    if p.kernel_sizes.is_empty() {
        return Err(Error::Config(
            "WaveNet Head: kernel_sizes must be non-empty".into(),
        ));
    }
    if p.kernel_sizes.contains(&0) {
        return Err(Error::Config(
            "WaveNet Head: kernel_sizes entries must be >= 1".into(),
        ));
    }
    Ok(())
}

impl Head {
    fn new(p: &PostStackHeadConfig, r: &mut WeightReader<'_>) -> Result<Self> {
        let n = p.kernel_sizes.len();
        let mut convs = Vec::with_capacity(n);
        let mut cin = p.in_channels;
        for (i, &k) in p.kernel_sizes.iter().enumerate() {
            let cout = if i + 1 == n {
                p.out_channels
            } else {
                p.channels
            };
            convs.push(Conv1D::new(cin, cout, k, true, 1, 1, r)?);
            cin = cout;
        }
        Ok(Self {
            convs,
            activation: p.activation.clone(),
            in_channels: p.in_channels,
        })
    }

    /// The reference's `receptive_field()`: one, plus `k - 1` per convolution.
    fn receptive_field(&self) -> usize {
        1 + self
            .convs
            .iter()
            .map(|c| c.kernel_size() - 1)
            .sum::<usize>()
    }

    fn set_max_buffer_size(&mut self, arena: &mut Arena, n: usize) {
        for c in &mut self.convs {
            c.set_max_buffer_size(arena, n);
        }
    }

    /// Activation then convolution, repeated. `work` holds the scaled head
    /// output on entry and is modified in place by the first activation.
    fn process(&mut self, arena: &mut Arena, work: &mut Buf, n: usize) {
        for i in 0..self.convs.len() {
            if i == 0 {
                let in_ch = self.convs[0].in_channels();
                self.activation.apply(&mut work.data_mut()[..in_ch * n]);
                self.convs[0].process(arena, &*work, n);
            } else {
                let (prev, cur) = self.convs.split_at_mut(i);
                let in_ch = cur[0].in_channels();
                self.activation
                    .apply(&mut prev[i - 1].output.data_mut()[..in_ch * n]);
                cur[0].process(arena, &prev[i - 1].output, n);
            }
        }
    }

    fn last_output(&self) -> &Buf {
        &self
            .convs
            .last()
            .expect("head has at least one conv")
            .output
    }

    fn has_cached_prewarm_state(&self) -> bool {
        self.convs.iter().all(Conv1D::has_cached_prewarm_state)
    }

    fn prewarm_from_cache(&mut self, arena: &mut Arena) {
        for c in &mut self.convs {
            c.prewarm_from_cache(arena);
        }
    }

    fn cache_state_as_prewarmed(&mut self, arena: &Arena) {
        for c in &mut self.convs {
            c.cache_prewarm_state(arena);
        }
    }
}

// ---------------------------------------------------------------------------
// WaveNet
// ---------------------------------------------------------------------------

/// A WaveNet model.
#[derive(Debug)]
pub(crate) struct WaveNet {
    layer_arrays: Vec<LayerArray>,
    post_stack_head: Option<Head>,
    /// The nested model that turns the input into the conditioning signal.
    ///
    /// The reference stores a whole nested `DSP` here (`condition_dsp` in the
    /// config), which is how a parametric capture reads knob positions: the
    /// outer WaveNet still sees the raw input as its layer input, but every
    /// layer's mixin and FiLM site is driven by this model's output instead.
    condition_dsp: Option<Box<dyn Engine>>,
    condition_input: Buf,
    condition_output: Buf,
    scaled_head_scratch: Buf,
    /// `(out_channels, max_buffer)`: where the core leaves its result, so
    /// that the slice-based and buffer-based entry points share one body.
    output_buf: Buf,
    head_scale: f32,
    in_channels: usize,
    out_channels: usize,
    prewarm_samples: usize,
    arena: Arena,
    max_buffer: usize,
    prewarm_on_reset: bool,
}

/// Everything about the shape that can be checked before a weight is read,
/// in the reference's order, so that a file with several problems reports
/// the one the reference reports.
fn check_shape(cfg: &WaveNetConfig, condition_dsp: Option<&dyn Engine>) -> Result<usize> {
    if cfg.layer_arrays.is_empty() {
        return Err(Error::Config(
            "WaveNet requires at least one layer array".into(),
        ));
    }
    let last_head_size = cfg.layer_arrays.last().expect("checked above").head_size;
    let out_channels = match &cfg.head {
        Some(h) => {
            if h.in_channels != last_head_size {
                return Err(Error::Config(format!(
                    "WaveNet head in_channels ({}) must match last layer array head_size ({last_head_size})",
                    h.in_channels
                )));
            }
            h.out_channels
        }
        None => last_head_size,
    };

    // The reference asserts both of these before reading a single weight.
    if let Some(dsp) = condition_dsp {
        if dsp.in_channels() != cfg.in_channels {
            return Err(Error::Config(format!(
                "input channels of WaveNet ({}) don't match input channels of condition DSP ({})",
                cfg.in_channels,
                dsp.in_channels()
            )));
        }
        for (i, p) in cfg.layer_arrays.iter().enumerate() {
            if p.condition_size != dsp.out_channels() {
                return Err(Error::Config(format!(
                    "condition_size of layer {i} ({}) doesn't match output channels of condition DSP ({})",
                    p.condition_size,
                    dsp.out_channels()
                )));
            }
        }
    }

    for (i, p) in cfg.layer_arrays.iter().enumerate() {
        if i > 0 && p.channels != cfg.layer_arrays[i - 1].head_size {
            return Err(Error::Config(format!(
                "channels of layer {i} ({}) doesn't match head_size of preceding layer ({})",
                p.channels,
                cfg.layer_arrays[i - 1].head_size
            )));
        }
        // Two more shapes have to line up between consecutive layer arrays,
        // and the reference checks neither. Array `i` is handed array
        // `i - 1`'s *layer* output, which is `channels` wide, while its
        // rechannel is built for `input_size`. When they disagree the reference walks
        // off the end of an Eigen block and trips an assertion inside
        // Eigen; with `NDEBUG` set it is undefined behaviour.
        let expected_input = if i == 0 {
            cfg.in_channels
        } else {
            cfg.layer_arrays[i - 1].channels
        };
        if p.input_size != expected_input {
            return Err(Error::Config(format!(
                "input_size of layer array {i} ({}) must be {expected_input}, the {} it is fed",
                p.input_size,
                if i == 0 {
                    "model's in_channels"
                } else {
                    "channels of the preceding array"
                }
            )));
        }
        // And array `i` seeds its skip accumulator by copying array `i - 1`'s
        // head output, so the two must be the same width.
        if i > 0 {
            let accumulator = head_output_size(p);
            let incoming = cfg.layer_arrays[i - 1].head_size;
            if accumulator != incoming {
                return Err(Error::Config(format!(
                    "layer array {i} accumulates {accumulator} skip channels ({}), but the \
                     preceding array's head produces {incoming}",
                    if p.head1x1.active {
                        "head1x1.out_channels"
                    } else {
                        "bottleneck"
                    }
                )));
            }
        }
        check_layer_array(p)?;
    }
    if let Some(h) = &cfg.head {
        check_head(h)?;
    }
    Ok(out_channels)
}

impl WaveNet {
    /// Build from the file's config and its flat weight array.
    ///
    /// The weights are consumed in the reference's order: every layer array in
    /// turn, then the post-stack head, then one final float for `head_scale`,
    /// which *overrides* the value in the JSON config.
    ///
    /// A `condition_dsp`, when the config carries one, is a complete nested
    /// model and is built first, because the outer model validates its
    /// channel counts before reading a single weight.
    ///
    /// Fails with [`Error::Config`] when the shapes do not fit together or
    /// the convolution histories would exceed [`MAX_HISTORY_FLOATS`] in
    /// total, and [`Error::WeightCount`] when `weights` is not exactly as
    /// long as the architecture consumes.
    pub(crate) fn new(mut cfg: WaveNetConfig, weights: &[f32]) -> Result<Self> {
        let condition_dsp: Option<Box<dyn Engine>> = match cfg.condition_dsp.take() {
            None => None,
            Some(nested) => Some(engine::build_file(*nested)?),
        };
        let out_channels = check_shape(&cfg, condition_dsp.as_deref())?;

        let mut r = WeightReader::new(weights);
        let layer_arrays = cfg
            .layer_arrays
            .iter()
            .map(|p| LayerArray::new(p, &mut r))
            .collect::<Result<Vec<_>>>()?;
        let post_stack_head = match &cfg.head {
            Some(h) => Some(Head::new(h, &mut r)?),
            None => None,
        };
        let head_scale = r.next()?;
        r.finish()?;

        // Every convolution's history has to fit before any is allocated.
        let mut history = 0usize;
        let convs = layer_arrays
            .iter()
            .flat_map(LayerArray::convs)
            .chain(post_stack_head.iter().flat_map(|h| h.convs.iter()));
        for c in convs {
            history = history.saturating_add(c.history_floats()?);
        }
        if history > MAX_HISTORY_FLOATS {
            return Err(Error::Config(format!(
                "the model's convolution histories total {history} floats; the supported maximum is {MAX_HISTORY_FLOATS}"
            )));
        }

        // The reference computes this once at construction: the condition
        // DSP's own prewarm (or one sample when there is none), plus each
        // array's receptive field, plus the head's minus one.
        let mut prewarm_samples = condition_dsp.as_ref().map_or(1, |d| d.prewarm_samples());
        for a in &layer_arrays {
            prewarm_samples = prewarm_samples.saturating_add(a.receptive_field()?);
        }
        if let Some(h) = &post_stack_head {
            prewarm_samples = prewarm_samples.saturating_add(h.receptive_field() - 1);
        }

        Ok(Self {
            layer_arrays,
            post_stack_head,
            condition_dsp,
            condition_input: Buf::new(),
            condition_output: Buf::new(),
            scaled_head_scratch: Buf::new(),
            output_buf: Buf::new(),
            head_scale,
            in_channels: cfg.in_channels,
            out_channels,
            prewarm_samples,
            arena: Arena::new(),
            max_buffer: 0,
            // `nam::DSP`'s constructor takes `gPrewarmOnResetDefault`, which
            // is true unless a host has scoped it off.
            prewarm_on_reset: true,
        })
    }

    /// The same model with every plain layer array on the fused path; see
    /// [`crate::fused`]. An array that is not plain, or is wider than that
    /// path takes, is left as it is, and so is every array of a model whose
    /// condition is more than one channel.
    ///
    /// For a model fresh from [`WaveNet::new`], before anything has sized
    /// it. The output is the same to the bit either way.
    pub(crate) fn fused(mut self) -> Self {
        // The condition is the input itself unless a nested model makes it.
        let condition_channels = match &self.condition_dsp {
            Some(dsp) => dsp.out_channels(),
            None => self.in_channels,
        };
        if condition_channels == 1 {
            let count = self.layer_arrays.len();
            for (i, a) in self.layer_arrays.iter_mut().enumerate() {
                a.fuse(i + 1 < count);
            }
        }
        self
    }

    /// How many of the layer arrays run fused.
    #[cfg(test)]
    pub(crate) fn fused_arrays(&self) -> usize {
        self.layer_arrays
            .iter()
            .filter(|a| matches!(a.layers, Layers::Fused(_)))
            .count()
    }

    /// Whether a steady state has been cached, letting a later reset skip the
    /// prewarm run entirely.
    fn has_cached_prewarm_state(&self) -> bool {
        // The reference refuses to cache when a condition DSP is present: its
        // state is outside the cache's reach, so restoring only the
        // convolution histories would not be a steady state at all.
        self.condition_dsp.is_none()
            && self
                .layer_arrays
                .iter()
                .all(LayerArray::has_cached_prewarm_state)
            && self
                .post_stack_head
                .as_ref()
                .is_none_or(Head::has_cached_prewarm_state)
    }

    /// Everything between the input and the output, with `condition_input`
    /// already filled and the result left in `output_buf`.
    fn run_core(&mut self, n: usize) {
        debug_assert!(
            n <= self.max_buffer,
            "block of {n} exceeds max buffer {}",
            self.max_buffer
        );
        let Self {
            condition_dsp,
            condition_input,
            condition_output,
            layer_arrays,
            post_stack_head,
            scaled_head_scratch,
            output_buf,
            arena,
            head_scale,
            ..
        } = self;

        match condition_dsp {
            // With no condition DSP the conditioning signal is the input itself.
            None => condition_output
                .left_mut(n)
                .copy_from_slice(condition_input.left(n)),
            Some(dsp) => dsp.process_buf(condition_input, condition_output, n),
        }

        for i in 0..layer_arrays.len() {
            if i == 0 {
                layer_arrays[0].process(arena, condition_input, condition_output, None, n);
            } else {
                let (prev, cur) = layer_arrays.split_at_mut(i);
                let p = &prev[i - 1];
                cur[0].process(
                    arena,
                    p.layer_outputs(),
                    condition_output,
                    Some(p.head_outputs()),
                    n,
                );
            }
        }

        let final_head = layer_arrays
            .last()
            .expect("checked at construction")
            .head_outputs();
        if let Some(head) = post_stack_head {
            let dst = scaled_head_scratch.left_mut(n);
            let src = final_head.left(n);
            for i in 0..head.in_channels * n {
                dst[i] = *head_scale * src[i];
            }
            head.process(arena, scaled_head_scratch, n);
            output_buf
                .left_mut(n)
                .copy_from_slice(head.last_output().left(n));
        } else {
            let dst = output_buf.left_mut(n);
            let src = final_head.left(n);
            for i in 0..dst.len() {
                dst[i] = *head_scale * src[i];
            }
        }
    }
}

/// What the A2 fast path needs from a model of that shape, lifted from the
/// built layers so the file is read once, one way. See [`crate::a2_fast`].
#[derive(Debug, Clone)]
pub(crate) struct A2Parts {
    /// Channels, which is also the bottleneck and the head's input width.
    pub channels: usize,
    /// The rechannel, `channels` weights from the one input.
    pub rechannel: Vec<f32>,
    /// The layers, in order.
    pub layers: Vec<A2LayerParts>,
    /// The head convolution's weights per tap, `channels` each, one output.
    pub head_w: Vec<Vec<f32>>,
    /// The head's bias.
    pub head_b: f32,
    /// The scale on the head's output.
    pub head_scale: f32,
    /// What the model reports for [`Engine::prewarm_samples`].
    pub prewarm_samples: usize,
}

/// One layer of [`A2Parts`].
#[derive(Debug, Clone)]
pub(crate) struct A2LayerParts {
    pub kernel_size: usize,
    pub dilation: usize,
    /// Per tap, column-major `(channels, channels)`.
    pub conv_w: Vec<Vec<f32>>,
    pub conv_b: Vec<f32>,
    /// The condition mixin, `channels` weights from the one condition.
    pub mixin_w: Vec<f32>,
    /// The 1x1 after the activation, column-major `(channels, channels)`.
    pub l1x1_w: Vec<f32>,
    pub l1x1_b: Vec<f32>,
    /// The leaky ReLU's slope below zero.
    pub slope: f32,
}

impl WaveNet {
    /// The parts of this model if it has the A2 shape: one layer array of
    /// plain, ungated, un-FiLMed layers with a leaky ReLU, a residual 1x1
    /// with bias and no head 1x1, a bottleneck equal to the channels, one
    /// input, one output, no condition model and no post-stack head.
    /// Anything else is `None`, and so is a model that has been
    /// [`WaveNet::fused`], whose layers are no longer there to read.
    pub(crate) fn a2_parts(&self) -> Option<A2Parts> {
        if self.layer_arrays.len() != 1
            || self.post_stack_head.is_some()
            || self.condition_dsp.is_some()
            || self.in_channels != 1
            || self.out_channels != 1
        {
            return None;
        }
        let a = &self.layer_arrays[0];
        let c = a.rechannel.out_channels();
        if c == 0
            || a.rechannel.in_channels() != 1
            || !a.rechannel.is_plain()
            || !a.rechannel.bias().is_empty()
        {
            return None;
        }
        let head = &a.head_rechannel;
        if head.in_channels() != c
            || head.out_channels() != 1
            || !head.is_plain()
            || head.dilation() != 1
            || head.bias().len() != 1
        {
            return None;
        }
        let Layers::Each(each) = &a.layers else {
            return None;
        };
        let mut layers = Vec::with_capacity(each.len());
        for l in each {
            let p = l.plain_parts(c)?;
            let Activation::LeakyReLU { negative_slope } = p.activation else {
                return None;
            };
            layers.push(A2LayerParts {
                kernel_size: p.kernel_size,
                dilation: p.dilation,
                conv_w: p.conv_w,
                conv_b: p.conv_b,
                mixin_w: p.mixin_w,
                l1x1_w: p.l1x1_w,
                l1x1_b: p.l1x1_b,
                slope: negative_slope,
            });
        }
        Some(A2Parts {
            channels: c,
            rechannel: a.rechannel.weight().to_vec(),
            layers,
            head_w: head.tap_matrices(),
            head_b: head.bias()[0],
            head_scale: self.head_scale,
            prewarm_samples: self.prewarm_samples,
        })
    }
}

impl Engine for WaveNet {
    fn in_channels(&self) -> usize {
        self.in_channels
    }

    fn out_channels(&self) -> usize {
        self.out_channels
    }

    fn prewarm_samples(&self) -> usize {
        self.prewarm_samples
    }

    fn max_buffer_size(&self) -> usize {
        self.max_buffer
    }

    /// Size every buffer for blocks of up to `max_buffer` frames.
    ///
    /// Allocating. The arena is rebuilt from scratch so histories stay packed
    /// in execution order.
    fn set_max_buffer_size(&mut self, max_buffer: usize) {
        self.max_buffer = max_buffer;
        self.condition_input.resize(self.in_channels, max_buffer);
        self.output_buf.resize(self.out_channels, max_buffer);
        match &mut self.condition_dsp {
            None => self.condition_output.resize(self.in_channels, max_buffer),
            Some(dsp) => {
                dsp.set_max_buffer_size(max_buffer);
                self.condition_output.resize(dsp.out_channels(), max_buffer);
            }
        }
        let Self {
            arena,
            layer_arrays,
            post_stack_head,
            scaled_head_scratch,
            ..
        } = self;
        arena.clear();
        for a in layer_arrays.iter_mut() {
            a.set_max_buffer_size(arena, max_buffer);
        }
        if let Some(h) = post_stack_head {
            h.set_max_buffer_size(arena, max_buffer);
            scaled_head_scratch.resize(h.in_channels, max_buffer);
        }
    }

    /// Record the prewarm-on-reset setting and propagate it to the
    /// conditioning model: `WaveNet::SetPrewarmOnReset`, which is
    /// `DSP::SetPrewarmOnReset` plus the hand-down.
    fn set_prewarm_on_reset(&mut self, on: bool) {
        self.prewarm_on_reset = on;
        if let Some(dsp) = &mut self.condition_dsp {
            dsp.set_prewarm_on_reset(on);
        }
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    /// Settle the model on silence, or restore the cached steady state.
    fn prewarm(&mut self) {
        if self.has_cached_prewarm_state() {
            let Self {
                arena,
                layer_arrays,
                post_stack_head,
                ..
            } = self;
            for a in layer_arrays.iter_mut() {
                a.prewarm_from_cache(arena);
            }
            if let Some(h) = post_stack_head {
                h.prewarm_from_cache(arena);
            }
            return;
        }

        let max_buffer = self.max_buffer;
        engine::prewarm_with_silence(self, max_buffer);

        let Self {
            arena,
            layer_arrays,
            post_stack_head,
            ..
        } = self;
        for a in layer_arrays.iter_mut() {
            a.cache_state_as_prewarmed(arena);
        }
        if let Some(h) = post_stack_head {
            h.cache_state_as_prewarmed(arena);
        }
    }

    /// Process `n` frames.
    ///
    /// Extra channels are ignored rather than trusted. The reference bounds
    /// both loops by the model, with `_set_condition_array` running to
    /// `NumInputChannels()` and the output copy to `out_channels`, and a
    /// host that hands over a stereo pair for a mono capture is common
    /// enough that the difference matters.
    fn process(&mut self, input: &[&[f32]], output: &mut [&mut [f32]], n: usize) {
        debug_assert!(input.len() >= self.in_channels);
        debug_assert!(output.len() >= self.out_channels);
        self.condition_input.copy_from_channels(input, n);
        self.run_core(n);
        self.output_buf.copy_to_channels(output, n);
    }

    fn process_buf(&mut self, input: &Buf, output: &mut Buf, n: usize) {
        // A flat copy of the first `n` columns is the top rows only when the
        // row counts agree, so the contract is equality rather than "at
        // least"; every caller in this crate hands over exactly shaped
        // buffers, and a mismatch fails on the slice lengths instead of
        // silently reading the wrong frames.
        debug_assert_eq!(input.rows(), self.in_channels);
        debug_assert_eq!(output.rows(), self.out_channels);
        self.condition_input
            .left_mut(n)
            .copy_from_slice(input.left(n));
        self.run_core(n);
        output.left_mut(n).copy_from_slice(self.output_buf.left(n));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{ArchConfig, parse_value};
    use serde_json::json;

    fn wavenet(dilations: Vec<u64>, kernel_size: u64, weights: usize) -> Result<WaveNet> {
        let file = parse_value(&json!({
            "version": "0.5.4", "architecture": "WaveNet", "sample_rate": 48000,
            "weights": vec![0.01f64; weights],
            "config": {"layers": [{
                "input_size": 1, "condition_size": 1, "head_size": 1, "channels": 2,
                "kernel_size": kernel_size, "dilations": dilations, "activation": "Tanh",
                "gated": false, "head_bias": false
            }], "head": null, "head_scale": 0.02}
        }))?;
        let ArchConfig::WaveNet(cfg) = file.config else {
            panic!()
        };
        WaveNet::new(cfg, &file.weights)
    }

    /// One 2-channel layer with kernel 3 and one dilation: rechannel 2,
    /// conv 12 + 2, mixin 2, 1x1 4 + 2, head rechannel 2, head_scale 1.
    const SMALL: usize = 2 + 14 + 2 + 6 + 2 + 1;

    #[test]
    fn the_weight_array_must_be_exactly_consumed() {
        assert!(wavenet(vec![1], 3, SMALL).is_ok());
        assert!(matches!(
            wavenet(vec![1], 3, SMALL - 1),
            Err(Error::WeightCount { .. })
        ));
        assert!(matches!(
            wavenet(vec![1], 3, SMALL + 1),
            Err(Error::WeightCount { .. })
        ));
    }

    #[test]
    fn a_shape_the_file_cannot_fill_is_refused_before_it_is_built() {
        // Kernel size 2^20 on two channels wants four million floats; the
        // file has thirty. The refusal names the shortfall rather than
        // allocating for it.
        match wavenet(vec![1], 1 << 20, 30) {
            Err(Error::WeightCount { expected, found }) => {
                assert!(expected > 1 << 20, "{expected}");
                assert_eq!(found, 30);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn histories_are_capped_in_total() {
        // A dilation of 2^20 on a 3-tap, 2-channel convolution is 4 M floats
        // of history: nine such layers exceed the cap, one does not.
        let nine = 2 + 9 * (14 + 2 + 6) + 2 + 1;
        let err = wavenet(vec![1 << 20; 9], 3, nine).unwrap_err().to_string();
        assert!(err.contains("convolution histories total"), "{err}");
        assert!(wavenet(vec![1 << 20], 3, SMALL).is_ok());
    }
}
