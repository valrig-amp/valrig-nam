#!/usr/bin/env bash
# Regenerate assets/ from the C++ reference implementation.
#
# Everything under assets/vectors/ is produced by *running the reference*, so
# the Rust test suite compares against real reference output rather than
# against itself. Run this when bumping NAM_REF_REV; commit the result.
#
#   ./tools/regen-assets.sh [workdir]
#
# Requires: git, a C++20 compiler, ~1 GB of disk for the Eigen checkout.
# assets/models/ is not touched: the four derived models there (seq, slim_bn,
# slim_cond, slim_pair) are hand-made from upstream's example_models, and the
# rest are copied from it verbatim.
set -euo pipefail

NAM_REF_REPO="${NAM_REF_REPO:-https://github.com/sdatkinson/NeuralAmpModelerCore.git}"
# The commit assets/ was produced from. Bump it deliberately, regenerate, and
# read the diff of assets/expectations.txt before committing.
NAM_REF_REV="${NAM_REF_REV:-2563c0fd4cb1f9ce457d89a761738ea15097e1f3}"
EIGEN_REPO="${EIGEN_REPO:-https://gitlab.com/libeigen/eigen.git}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE="$(dirname "$HERE")"
WORK="${1:-${TMPDIR:-/tmp}/valverig-nam-reference}"
ASSETS="$CRATE/assets"

mkdir -p "$WORK"
if [ ! -d "$WORK/nam-ref/.git" ]; then
  git clone "$NAM_REF_REPO" "$WORK/nam-ref"
fi
git -C "$WORK/nam-ref" checkout --quiet "$NAM_REF_REV"
if [ ! -d "$WORK/nam-ref/Dependencies/eigen/Eigen" ]; then
  rmdir "$WORK/nam-ref/Dependencies/eigen" 2>/dev/null || true
  git clone --depth 1 "$EIGEN_REPO" "$WORK/nam-ref/Dependencies/eigen"
fi

REF="$WORK/nam-ref"
cp "$HERE/gen_vectors.cpp" "$REF/tools/gen_vectors.cpp"
cp "$HERE/gen_activations.cpp" "$REF/tools/gen_activations.cpp"

# The build configuration IS part of the contract. See README.md:
#   -ffp-contract=off     no fused multiply-add, so a*b+c rounds twice as in Rust
#   EIGEN_DONT_VECTORIZE  scalar Eigen kernels, so reductions run in index order
#   no -ffast-math        no reassociation
mkdir -p "$REF/gbuild"
build() { # build <name> <extra flags...>
  local name="$1"; shift
  echo "building reference: $name"
  c++ -std=c++20 -I"$REF" -I"$REF/Dependencies/eigen" -I"$REF/Dependencies/nlohmann" -w "$@" \
      "$REF"/NAM/*.cpp "$REF"/NAM/wavenet/*.cpp "$REF/tools/gen_vectors.cpp" \
      -o "$REF/gbuild/ref_$name"
}

# The canonical build: its output is what assets/vectors/*.out.f64 contains
# and what valverig-nam is compared against.
build canonical -O2 -ffp-contract=off -DEIGEN_DONT_VECTORIZE -DEIGEN_MAX_ALIGN_BYTES=0

# Three other configurations the reference itself supports and ships. They are
# built only to measure how far the reference drifts from *itself*, which is
# the yardstick assets/reference-spread.txt records.
build inline  -O2 -ffp-contract=off -DEIGEN_DONT_VECTORIZE -DEIGEN_MAX_ALIGN_BYTES=0 -DNAM_USE_INLINE_GEMM
build vec     -O3
build release -Ofast -DNAM_ENABLE_A2_FAST

echo "regenerating assets/activations.f32..."
c++ -std=c++20 -O2 -ffp-contract=off -DEIGEN_DONT_VECTORIZE -DEIGEN_MAX_ALIGN_BYTES=0 \
    -I"$REF" -I"$REF/Dependencies/eigen" -I"$REF/Dependencies/nlohmann" -w \
    "$REF/tools/gen_activations.cpp" "$REF"/NAM/activations.cpp \
    -o "$REF/gbuild/gen_activations"
"$REF/gbuild/gen_activations" "$ASSETS/activations.f32"

mkdir -p "$ASSETS/vectors"
gen() { # gen <model> <case-name> <n> <seed> <schedule> [extra flags...]
  local model="$1" name="$2"; shift 2
  echo "  $name"
  "$REF/gbuild/ref_canonical" "$ASSETS/models/$model" "$ASSETS/vectors/$name" "$@"
}

echo "generating vectors..."
for m in "$ASSETS"/models/*.nam; do
  b="$(basename "$m" .nam)"
  # Block schedules chosen to exercise every ring-buffer path: a steady
  # power-of-two block, a ragged schedule that straddles rewinds, and
  # single-sample calls.
  gen "$b.nam" "${b}__b64"      2048 20240501 64
  gen "$b.nam" "${b}__ragged"   2048 20240501 1,7,64,3,128,17
  gen "$b.nam" "${b}__b1"        512 20240501 1
  gen "$b.nam" "${b}__noprewarm" 2048 20240501 64 --no-prewarm
  # Stress cases: the maximum block equal to one frame (so prewarm runs one
  # frame at a time), a maximum far larger than any real host uses, and a
  # schedule of primes that never lines up with any internal buffer boundary.
  gen "$b.nam" "${b}__b1max"      600 777 1
  gen "$b.nam" "${b}__bigblock"  4096 777 4096,1,4095,2
  gen "$b.nam" "${b}__prime"     2003 777 13,29,101,7,257
done

# Size-switchable models additionally get one vector per size. Two things have
# to hold: the reference must implement the interface (asked, not hard-coded -
# `--slim` fails otherwise), and valverig-nam must implement it too. Only
# `SlimmableContainer` does. A slimmable *WaveNet* answers the reference's
# probe but is run here at full width only, so generating its narrower widths
# would record vectors nothing checks.
for m in "$ASSETS"/models/*.nam; do
  b="$(basename "$m" .nam)"
  grep -q '"architecture"[[:space:]]*:[[:space:]]*"SlimmableContainer"' "$m" || continue
  "$REF/gbuild/ref_canonical" "$m" "$WORK/slimprobe" 8 1 8 --slim 0.5 >/dev/null 2>&1 || continue
  for v in 0.0 0.25 0.33 0.5 0.75 1.0; do
    tag="$(echo "$v" | tr . _)"
    gen "$b.nam" "${b}__slim$tag" 1024 20240501 64 --slim "$v"
  done
done

# Measure the reference's own build-to-build spread, case by case. valverig-nam is
# held to this: where the reference agrees with itself we must match it exactly,
# and where it does not we must land inside the envelope it spans.
echo "measuring the reference's build-to-build spread..."
SPREAD="$ASSETS/reference-spread.txt"
{
  echo "# Largest disagreement among the reference's own supported build"
  echo "# configurations, per case, relative to the signal peak:"
  echo "#   canonical  -O2, scalar Eigen, -ffp-contract=off  (what assets/vectors holds)"
  echo "#   inline     the same, plus -DNAM_USE_INLINE_GEMM"
  echo "#   vec        -O3, vectorised Eigen"
  echo "#   release    -Ofast, -DNAM_ENABLE_A2_FAST  (the CMake Release default)"
  echo "# Regenerated by tools/regen-assets.sh. See README.md, \"How close, and to what\"."
  echo
  for meta in "$ASSETS"/vectors/*.meta.json; do
    name="$(basename "$meta" .meta.json)"
    model="${name%%__*}"
    # Re-run the same case under each configuration into a scratch directory.
    tmp="$WORK/spread/$name"; mkdir -p "$tmp"
    args="$(sed -n "s/.*\"num_samples\":\([0-9]*\).*/\1/p" "$meta")"
    sched="$(sed -n "s/.*\"schedule\":\[\([0-9,]*\)\].*/\1/p" "$meta")"
    seed=20240501
    case "$name" in *__b1max|*__bigblock|*__prime) seed=777 ;; esac
    pw=""; grep -q '"prewarm":false' "$meta" && pw="--no-prewarm"
    extra="$pw"
    case "$name" in
      *__slim*) extra="$extra --slim $(echo "${name##*__slim}" | tr _ .)" ;;
    esac
    outs=""
    for cfg in canonical inline vec release; do
      # shellcheck disable=SC2086
      "$REF/gbuild/ref_$cfg" "$ASSETS/models/$model.nam" "$tmp/$cfg" \
        "$args" "$seed" "$sched" $extra >/dev/null 2>&1 || continue
      outs="$outs $tmp/$cfg.out.f64"
    done
    # shellcheck disable=SC2086
    val="$("$REF/gbuild/ref_canonical" --spread $outs 2>/dev/null)"
    printf "%s %s\n" "$name" "${val:-n/a}"
  done
} > "$SPREAD"

echo "done. $(ls "$ASSETS/vectors" | wc -l | tr -d ' ') files in assets/vectors/"
echo "      $(grep -vc '^#' "$SPREAD" 2>/dev/null || echo 0) entries in assets/reference-spread.txt"
