# valrig-nam

Pure-Rust [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler)
inference. Loads `.nam` captures and runs them allocation-free in an audio
callback.

```rust
use valrig_nam::loader::Model;

let mut model = Model::from_file("captures/plexi.nam")?;

// Allocates and settles the model. Before the audio thread starts.
model.reset(48_000.0, 64);

// Allocation-free from here on, for any block of up to 64 frames.
let mut wet = [0.0f32; 64];
model.process_mono(&dry, &mut wet);
```

A capture is only correct at the sample rate it was trained at. So nothing will
resample. You can compare `Model::expected_sample_rate` against the stream and handle
as you want.

## How to use

`process()` does not allocate, free, lock if `reset()` has been called once.
So it's safe to call from any audio callback. Any call before `reset` panics.
Then, there is no error path out of an audio callback.

`reset()` and `set_slimmable_size()` should then be called outside of an audio thread.

## Accuracy

"Bit-exact against the C++ reference" is not the taken assumption because
it deeply depends on the building phase. `NeuralAmpModelerCore` does its linear
algebra with Eigen, whose reduction order and use of fused multiply-adds
change with vectorisation settings, optimisation level and the reference's
own compile-time paths. This makes it impossible to predict in advance what would
be the end result by up to 4e-6 relative to the signal peak (~ -108 dBFS).

The current implementation replicates as close as possible the result of the
original one. The `assets/vectors/` holds input/output pairs produced by running
the real reference. `assets/expectations.txt` records how far this crate may sit,
and `tests/reference.rs` additionally bounds every case at twice the reference's
own build-to-build spread.

## Coverage

Every architecture the reference registers:

| architecture | status |
|---|---|
| WaveNet, all generations: legacy `gated`, `bottleneck`, `groups`, per-layer `kernel_sizes` and activations, `head1x1`, `layer1x1`, the eight FiLM sites, gated and blended modes, the nested `head` object, the post-stack head | ✓ |
| LSTM | ✓ |
| `condition_dsp`, the nested model behind parametric captures | ✓ |
| `SlimmableContainer`, `Sequential` | ✓ |
| Slimmable WaveNet | read and run at full width |
| Linear, ConvNet | recognised, refused |

`.nam` versions 0.5.0 through 0.7.0, and every activation the reference
registers.

## Tests

```bash
cargo test --release    # about ten seconds
cargo test              # the same in debug, where the reference comparison takes minutes
```

`assets/` is committed, so nothing needs building or fetching. Regenerating
it needs a reference checkout, which the one script in `tools/` makes:

```bash
./tools/regen-assets.sh
```

One test measures rather than checks, how fast each bundled capture runs
against real time, so it is ignored by default:

```bash
cargo test --release --test benchmark -- --ignored --nocapture
```

## Licence

MIT; see [`LICENSE`](LICENSE). The `.nam` files under `assets/models/` are
test fixtures from
[NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)
(MIT, © 2023 Steven Atkinson), four of them modified. No reference code is
vendored.
