import { performance } from 'node:perf_hooks';

// Discord receive PCM and the playback decoder both use mono s16le at 16 kHz.
const SAMPLES_PER_MS = 16;
const WINDOW = 80 * SAMPLES_PER_MS;
const COARSE = 16;
const SEARCH_INTERVAL = 80 * SAMPLES_PER_MS;
const TAPS = 16;
const TAP_CENTER = 7;
// SILK microphone re-encoding can change a replay's waveform substantially.
// These gates prefer rejecting a confirmed replay over recognizing words from
// it. Quiet speech buried under the same loud playback can be lost; this is
// deliberately not advertised as general-purpose acoustic source separation.
const MAX_CODEC_RESIDUAL = 0.55;
const MAX_HELDOUT_RESIDUAL = 0.65;
const MAX_SPECTRAL_NOVELTY = 0.25;
const FFT_SIZE = 2048;
const FFT_COS = Float64Array.from({ length: FFT_SIZE / 2 }, (_, i) => Math.cos(-2 * Math.PI * i / FFT_SIZE));
const FFT_SIN = Float64Array.from({ length: FFT_SIZE / 2 }, (_, i) => Math.sin(-2 * Math.PI * i / FFT_SIZE));
const EMPTY = Buffer.alloc(0);

export interface EchoResult { pcm: Buffer; echoOnly: boolean; echoRelated?: boolean }
interface Span { start: number; end: number; at: number }
interface Snapshot {
  start: number;
  end: number;
  raw: Float32Array;
  coarse: Float64Array;
  sums: Float64Array;
  squares: Float64Array;
  spans: readonly Span[];
  breaks: readonly number[];
}
interface Match { start: number; score: number; gain: number }
interface LinearModel { taps: Float64Array; offset: number; at: number }

/** A bounded reference of audio actually consumed by the outgoing player.
 *
 * `now` is the packet observation time, not a file/decoder-buffer timestamp.
 * This is a conservative, linear reference canceller, not a room acoustic AEC.
 */
export class PlaybackEchoReference {
  readonly maxDelayMs: number;
  private readonly samples: Float32Array;
  private written = 0;
  private spans: Span[] = [];
  private breaks: number[] = [];
  private cached: Snapshot | null = null;
  private epoch = 0;

  constructor(options: { maxDelayMs?: number } = {}) {
    const delay = options.maxDelayMs ?? 2200;
    if (!Number.isFinite(delay) || delay < 100 || delay > 5000) throw new RangeError('Invalid echo delay');
    this.maxDelayMs = delay;
    this.samples = new Float32Array(Math.ceil((delay + 240) * SAMPLES_PER_MS / COARSE) * COARSE);
  }

  get generation(): number { return this.epoch; }

  append(pcm: Buffer, now = performance.now()): void {
    if (!pcm.length || pcm.length % 2 || !Number.isFinite(now)) return;
    const count = pcm.length / 2;
    const previous = this.spans.at(-1);
    if (previous && now < previous.at) this.clear();
    else if (previous && now - previous.at > count / SAMPLES_PER_MS + 60) this.breaks.push(this.written);
    if (!this.written) {
      // Align a quiet leading microphone interval with the first observed
      // playback onset. This zero prefix can never explain nonzero input.
      this.samples.fill(0, 0, WINDOW);
      this.written = WINDOW;
      this.spans.push({ start: 0, end: WINDOW, at: now - count / SAMPLES_PER_MS });
    }
    const start = this.written;
    // Only the bounded suffix is retained, even if a caller supplies a big block.
    for (let i = Math.max(0, count - this.samples.length); i < count; i++) {
      this.samples[(start + i) % this.samples.length] = pcm.readInt16LE(i * 2);
    }
    this.written += count;
    this.spans.push({ start, end: this.written, at: now });
    const oldest = this.written - this.samples.length;
    this.spans = this.spans.filter(span => span.end > oldest);
    this.breaks = this.breaks.filter(index => index > oldest);
    this.cached = null;
  }

  clear(): void {
    this.written = 0; this.spans = []; this.breaks = []; this.cached = null; this.epoch++;
  }

  /** @internal Shared immutable snapshot; eight receivers do not copy the ring eight times. */
  snapshot(now: number): Snapshot | null {
    const last = this.spans.at(-1);
    if (!last || now - last.at > this.maxDelayMs || now + 60 < last.at) return null;
    if (this.cached) return this.cached;
    const start = Math.ceil(Math.max(0, this.written - this.samples.length) / COARSE) * COARSE;
    const raw = new Float32Array(this.written - start);
    for (let i = 0; i < raw.length; i++) raw[i] = this.samples[(start + i) % this.samples.length]!;
    const coarse = new Float64Array(Math.floor(raw.length / COARSE));
    const sums = new Float64Array(coarse.length + 1), squares = new Float64Array(coarse.length + 1);
    for (let i = 0; i < coarse.length; i++) {
      let sum = 0;
      for (let j = 0; j < COARSE; j++) sum += raw[i * COARSE + j]!;
      const value = coarse[i] = sum / COARSE;
      sums[i + 1] = sums[i]! + value; squares[i + 1] = squares[i]! + value * value;
    }
    return this.cached = { start, end: this.written, raw, coarse, sums, squares,
      spans: this.spans.slice(), breaks: this.breaks.slice() };
  }
}

function centered(samples: Float32Array | Float64Array): { mean: number; energy: number } {
  let sum = 0, square = 0;
  for (const value of samples) { sum += value; square += value * value; }
  const mean = sum / samples.length;
  return { mean, energy: Math.max(0, square - sum * mean) };
}

function spectrum(samples: Float64Array): Float64Array {
  const real = new Float64Array(FFT_SIZE), imaginary = new Float64Array(FFT_SIZE);
  for (let i = 0; i < samples.length; i++) real[i] = samples[i]! * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (samples.length - 1)));
  for (let i = 1, j = 0; i < FFT_SIZE; i++) {
    let bit = FFT_SIZE >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { const value = real[i]!; real[i] = real[j]!; real[j] = value; }
  }
  for (let width = 2; width <= FFT_SIZE; width *= 2) {
    const half = width / 2, stride = FFT_SIZE / width;
    for (let first = 0; first < FFT_SIZE; first += width) {
      for (let j = 0; j < half; j++) {
        const a = first + j, b = a + half;
        const cosine = FFT_COS[j * stride]!, sine = FFT_SIN[j * stride]!;
        const re = real[b]! * cosine - imaginary[b]! * sine;
        const im = real[b]! * sine + imaginary[b]! * cosine;
        real[b] = real[a]! - re; imaginary[b] = imaginary[a]! - im;
        real[a] = real[a]! + re; imaginary[a] = imaginary[a]! + im;
      }
    }
  }
  return Float64Array.from({ length: FFT_SIZE / 2 }, (_, i) => real[i]! ** 2 + imaginary[i]! ** 2);
}

function hasNovelVoice(residual: Buffer, predict: (index: number) => number): boolean {
  const samples = residual.length / 2;
  const remaining = new Float64Array(samples), expected = new Float64Array(samples);
  for (let i = 0; i < samples; i++) { remaining[i] = residual.readInt16LE(i * 2); expected[i] = predict(i); }
  const noise = spectrum(remaining), reference = spectrum(expected);
  let maximum = 0, total = 0;
  for (let i = 2; i < reference.length; i++) { maximum = Math.max(maximum, reference[i]!); total += reference[i]!; }
  let novel = 0;
  for (let i = 2; i < reference.length; i++) {
    // Reject a substantial independent spectral component. This echo-biased
    // gate cannot recover quiet voices masked by the same playback bands.
    const allowance = Math.max(reference[i - 1]!, reference[i]!, reference[Math.min(i + 1, reference.length - 1)]!) * 0.3 ** 2;
    novel += Math.max(0, noise[i]! - allowance - maximum * 0.000001);
  }
  return !total || novel > total * MAX_SPECTRAL_NOVELTY;
}

function coupledResidual(residual: Buffer, snapshot: Snapshot, start: number): boolean {
  let sum = 0, peak = 0, chunks = 0;
  for (let first = 0; first < residual.length / 2; first += 320) {
    const length = Math.min(320, residual.length / 2 - first);
    if (length < 160) continue;
    let actualSum = 0, actualSquare = 0;
    for (let i = 0; i < length; i++) {
      const value = residual.readInt16LE((first + i) * 2);
      actualSum += value; actualSquare += value * value;
    }
    const energy = actualSquare - actualSum * actualSum / length;
    if (energy < length * 16) continue;
    let strongest = 0;
    for (let shift = -128; shift <= 128; shift += 8) {
      const at = start + first + shift - snapshot.start;
      if (at < 0 || at + length > snapshot.raw.length) continue;
      let expectedSum = 0, expectedSquare = 0, dot = 0;
      for (let i = 0; i < length; i++) {
        const value = snapshot.raw[at + i]!;
        expectedSum += value; expectedSquare += value * value;
        dot += residual.readInt16LE((first + i) * 2) * value;
      }
      const referenceEnergy = expectedSquare - expectedSum * expectedSum / length;
      dot -= actualSum * expectedSum / length;
      if (referenceEnergy > length * 16) strongest = Math.max(strongest, Math.abs(dot) / Math.sqrt(energy * referenceEnergy));
    }
    chunks++; sum += strongest; peak = Math.max(peak, strongest);
  }
  // Residual codec harmonics remain temporally related to the source. This
  // extra evidence avoids treating unrelated, similarly coloured noise as echo.
  return chunks > 0 && peak > 0.32 && sum / chunks > 0.25;
}

/** One instance per human receive stream. Never share alignment between users. */
export class PlaybackEchoFilter {
  private history = new Float32Array(0);
  private total = 0;
  private offset: number | null = null;
  private generation: number;
  private searchedAt = -Infinity;
  private pending: Buffer[] = [];
  private pendingBytes = 0;
  private model: LinearModel | null = null;
  private lastAt: number | null = null;

  constructor(private readonly reference: PlaybackEchoReference) { this.generation = reference.generation; }

  reset(): void {
    this.history = new Float32Array(0); this.total = 0; this.offset = null; this.searchedAt = -Infinity;
    this.pending = []; this.pendingBytes = 0; this.model = null; this.lastAt = null;
    this.generation = this.reference.generation;
  }

  process(pcm: Buffer, now = performance.now()): EchoResult {
    if (!pcm.length || pcm.length % 2 || !Number.isFinite(now)) return { pcm, echoOnly: false };
    // Discord's largest accepted packet is 60 ms. Bound the public API too.
    if (pcm.length > 120 * SAMPLES_PER_MS * 2) {
      const tail = this.takePending(); this.reset();
      return { pcm: tail.length ? Buffer.concat([tail, pcm]) : pcm, echoOnly: false };
    }
    if (this.generation !== this.reference.generation || (this.lastAt !== null && (now < this.lastAt || now - this.lastAt > 250))) {
      const tail = this.takePending(); this.reset();
      // A discontinuity is not contiguous audio: never fit the old tail to a
      // new reference. Drain it unchanged and start acquisition next packet.
      if (tail.length) return { pcm: Buffer.concat([tail, pcm]), echoOnly: false };
    }
    this.lastAt = now;
    const snapshot = this.reference.snapshot(now);
    if (!snapshot) {
      const tail = this.takePending(); this.offset = null; this.model = null;
      return { pcm: tail.length ? Buffer.concat([tail, pcm]) : pcm, echoOnly: false };
    }
    // Match complete bounded blocks, including after silence. No future
    // coherent echo is allowed to justify removing an earlier human prefix.
    this.pending.push(pcm); this.pendingBytes += pcm.length;
    if (this.pendingBytes < WINDOW * 2) return { pcm: EMPTY, echoOnly: false };
    const results: EchoResult[] = [];
    while (this.pendingBytes >= WINDOW * 2) {
      const block = this.takePending(WINDOW * 2);
      this.remember(block);
      results.push(this.clean(block, snapshot, now - this.pendingBytes / 32, false));
    }
    return { pcm: Buffer.concat(results.map(result => result.pcm)), echoOnly: results.every(result => result.echoOnly),
      echoRelated: results.some(result => result.echoRelated) };
  }

  /** Drain only on normal input end. Cancellation uses reset() and discards the tail. */
  flush(now = performance.now()): EchoResult {
    const pcm = this.takePending();
    if (!pcm.length) return { pcm: EMPTY, echoOnly: false };
    const snapshot = this.reference.snapshot(now);
    if (!snapshot || this.generation !== this.reference.generation) return { pcm, echoOnly: false };
    this.remember(pcm);
    return this.clean(pcm, snapshot, now, this.history.length < WINDOW);
  }

  private takePending(bytes = this.pendingBytes): Buffer {
    if (!bytes) return EMPTY;
    const all = Buffer.concat(this.pending, this.pendingBytes);
    const result = all.subarray(0, bytes);
    this.pendingBytes -= bytes;
    this.pending = this.pendingBytes ? [all.subarray(bytes)] : [];
    return result;
  }

  private remember(pcm: Buffer): void {
    const count = pcm.length / 2;
    const history = new Float32Array(count);
    for (let i = 0; i < count; i++) history[i] = pcm.readInt16LE(i * 2);
    this.history = history; this.total += count;
  }

  private valid(snapshot: Snapshot, start: number, length: number, now: number): boolean {
    const end = start + length;
    if (start < snapshot.start || end > snapshot.end) return false;
    if (snapshot.breaks.some(index => index > start && index < end)) return false;
    const span = snapshot.spans.find(value => end > value.start && end <= value.end);
    if (!span) return false;
    const at = span.at - (span.end - end) / SAMPLES_PER_MS;
    return at <= now + 20 && now - at <= this.reference.maxDelayMs;
  }

  private correlate(snapshot: Snapshot, start: number, mean: number, energy: number): Match | null {
    const index = start - snapshot.start;
    if (index < 0 || index + this.history.length > snapshot.raw.length) return null;
    let sum = 0, square = 0, dot = 0;
    for (let i = 0; i < this.history.length; i++) {
      const value = snapshot.raw[index + i]!;
      sum += value; square += value * value; dot += (this.history[i]! - mean) * value;
    }
    const referenceEnergy = square - sum * sum / this.history.length;
    if (referenceEnergy < this.history.length * 16) return null;
    const gain = dot / referenceEnergy;
    if (Math.abs(gain) < 0.005 || Math.abs(gain) > 8) return null;
    return { start, score: dot * dot / (energy * referenceEnergy), gain };
  }

  private match(snapshot: Snapshot, now: number, short: boolean): Match | null {
    const { mean, energy } = centered(this.history);
    if (this.history.length < 160 || energy < this.history.length * 16) return null;
    const threshold = short ? 0.98 ** 2 : 0.68 ** 2;
    let best: Match | null = null;
    const inspect = (start: number, radius: number): void => {
      for (let step = -radius; step <= radius; step++) {
        const candidate = this.correlate(snapshot, start + step, mean, energy);
        if (candidate && (!best || candidate.score > best.score)
          && this.valid(snapshot, candidate.start, this.history.length, now)) best = candidate;
      }
    };
    if (this.offset !== null) inspect(this.total - this.history.length + this.offset, 8);
    const tracked = best as Match | null;
    if ((!tracked || tracked.score < threshold) && (short || this.total - this.searchedAt >= SEARCH_INTERVAL)) {
      this.searchedAt = this.total;
      const length = Math.floor(this.history.length / COARSE);
      const input = new Float64Array(length);
      for (let i = 0; i < length; i++) {
        let sum = 0;
        for (let j = 0; j < COARSE; j++) sum += this.history[i * COARSE + j]!;
        input[i] = sum / COARSE;
      }
      const stats = centered(input);
      const peaks: { start: number; score: number }[] = [];
      if (stats.energy > 0.01) {
        for (let start = 0; start + length <= snapshot.coarse.length; start++) {
          const sum = snapshot.sums[start + length]! - snapshot.sums[start]!;
          const square = snapshot.squares[start + length]! - snapshot.squares[start]!;
          const variance = square - sum * sum / length;
          if (variance < 0.01) continue;
          let dot = 0;
          for (let i = 0; i < length; i++) dot += (input[i]! - stats.mean) * snapshot.coarse[start + i]!;
          const score = dot * dot / (stats.energy * variance);
          if (score < 0.12 || (peaks.length >= 4 && score <= peaks[peaks.length - 1]!.score)) continue;
          const nearby = peaks.findIndex(peak => Math.abs(peak.start - start) <= 3);
          if (nearby >= 0) { if (score <= peaks[nearby]!.score) continue; peaks.splice(nearby, 1); }
          peaks.push({ start, score }); peaks.sort((a, b) => b.score - a.score);
          if (peaks.length > 4) peaks.pop();
        }
      }
      for (const peak of peaks) inspect(snapshot.start + peak.start * COARSE, COARSE);
    }
    const found = best as Match | null;
    if (!found || found.score < threshold) return null;
    this.offset = found.start - (this.total - this.history.length);
    return found;
  }

  private tonal(snapshot: Snapshot, match: Match): boolean {
    // A lone tone has many apparently matching lags. It requires a held-out
    // waveform match after filtering, not just similar pitch/correlation.
    const start = match.start - snapshot.start;
    let predictor = 0, predictionEnergy = 0;
    for (let i = 2; i < this.history.length; i++) {
      const previous = snapshot.raw[start + i - 1]!;
      predictor += previous * (snapshot.raw[start + i]! + snapshot.raw[start + i - 2]!);
      predictionEnergy += previous * previous;
    }
    const coefficient = predictionEnergy ? predictor / predictionEnergy : 0;
    let error = 0, referenceEnergy = 0;
    for (let i = 2; i < this.history.length; i++) {
      const value = snapshot.raw[start + i]!;
      const difference = value - coefficient * snapshot.raw[start + i - 1]! + snapshot.raw[start + i - 2]!;
      error += difference * difference; referenceEnergy += value * value;
    }
    return Boolean(referenceEnergy && error / referenceEnergy < 0.0004);
  }

  private predict(snapshot: Snapshot, at: number, taps: Float64Array): number {
    let value = 0;
    for (let tap = 0; tap < taps.length; tap++) {
      const index = at + tap - TAP_CENTER - snapshot.start;
      if (index >= 0 && index < snapshot.raw.length) value += taps[tap]! * snapshot.raw[index]!;
    }
    return value;
  }

  private trainingEnd(): number {
    // A delayed onset can leave most of the block quiet. Split the signal's
    // energy rather than training only on silence and withholding all speech.
    let total = 0;
    for (const value of this.history) total += value * value;
    let seen = 0;
    for (let i = 0; i < this.history.length; i++) {
      seen += this.history[i]! ** 2;
      if (seen >= total * 0.6) return Math.min(this.history.length - TAPS * 4, Math.max(TAPS * 4, i));
    }
    return Math.floor(this.history.length * 0.6);
  }

  private fit(snapshot: Snapshot, match: Match): LinearModel | null {
    const offset = match.start - (this.total - this.history.length);
    if (this.model && this.model.offset === offset && this.total - this.model.at < SEARCH_INTERVAL) return this.model;
    const train = this.trainingEnd();
    const matrix = Array.from({ length: TAPS }, () => new Float64Array(TAPS));
    const right = new Float64Array(TAPS), values = new Float64Array(TAPS);
    let rows = 0;
    for (let i = TAPS; i < train; i += 2) {
      const at = match.start + i - snapshot.start;
      if (at - TAP_CENTER < 0 || at + TAPS - TAP_CENTER > snapshot.raw.length) continue;
      for (let tap = 0; tap < TAPS; tap++) values[tap] = snapshot.raw[at + tap - TAP_CENTER]!;
      for (let a = 0; a < TAPS; a++) {
        right[a] = right[a]! + values[a]! * this.history[i]!;
        for (let b = 0; b <= a; b++) matrix[a]![b] = matrix[a]![b]! + values[a]! * values[b]!;
      }
      rows++;
    }
    if (rows < TAPS * 3) return null;
    let scale = 0;
    for (let i = 0; i < TAPS; i++) scale += matrix[i]![i]!;
    const ridge = scale / TAPS * 0.0001;
    if (ridge < 0.01) return null;
    // Cholesky solve with ridge regularisation; only sixteen bounded taps.
    for (let i = 0; i < TAPS; i++) {
      for (let j = 0; j <= i; j++) {
        let value = matrix[i]![j]! + (i === j ? ridge : 0);
        for (let k = 0; k < j; k++) value -= matrix[i]![k]! * matrix[j]![k]!;
        if (i === j) { if (value <= 0) return null; matrix[i]![j] = Math.sqrt(value); }
        else matrix[i]![j] = value / matrix[j]![j]!;
      }
    }
    const intermediate = new Float64Array(TAPS), taps = new Float64Array(TAPS);
    for (let i = 0; i < TAPS; i++) {
      let value = right[i]!;
      for (let j = 0; j < i; j++) value -= matrix[i]![j]! * intermediate[j]!;
      intermediate[i] = value / matrix[i]![i]!;
    }
    for (let i = TAPS - 1; i >= 0; i--) {
      let value = intermediate[i]!;
      for (let j = i + 1; j < TAPS; j++) value -= matrix[j]![i]! * taps[j]!;
      taps[i] = value / matrix[i]![i]!;
    }
    if (taps.some(value => !Number.isFinite(value)) || taps.reduce((sum, value) => sum + Math.abs(value), 0) > 16) return null;
    let fitError = 0, scalarError = 0;
    // No fitting on the final 40% of signal energy: simultaneous speech cannot be
    // declared echo merely because a flexible filter fit its training samples.
    for (let i = train; i < this.history.length; i++) {
      const actual = this.history[i]!;
      fitError += (actual - this.predict(snapshot, match.start + i, taps)) ** 2;
      scalarError += (actual - match.gain * snapshot.raw[match.start - snapshot.start + i]!) ** 2;
    }
    if (fitError > Math.max(1, scalarError * 1.05)) return null;
    return this.model = { taps, offset, at: this.total };
  }

  private clean(pcm: Buffer, snapshot: Snapshot, now: number, short: boolean): EchoResult {
    const match = this.match(snapshot, now, short);
    if (!match) {
      // A brief final packet has too little evidence to cancel confidently.
      // Still report an already established, correlated path so the receiver
      // does not treat a 20 ms codec tail as an independent spoken command.
      if (short && this.offset !== null) {
        const { mean, energy } = centered(this.history);
        const candidate = energy > this.history.length * 16
          ? this.correlate(snapshot, this.total - this.history.length + this.offset, mean, energy) : null;
        if (candidate && candidate.score >= 0.68 ** 2 && this.valid(snapshot, candidate.start, this.history.length, now)) {
          return { pcm, echoOnly: false, echoRelated: true };
        }
      }
      return { pcm, echoOnly: false };
    }
    const count = pcm.length / 2;
    const start = match.start;
    if (!this.valid(snapshot, start, count, now)) return { pcm, echoOnly: false };
    const model = this.fit(snapshot, match);
    const predicted = (at: number): number => model ? this.predict(snapshot, at, model.taps)
      : match.gain * snapshot.raw[at - snapshot.start]!;
    let verifiedError = 0, verifiedEnergy = 0;
    for (let i = this.trainingEnd(); i < this.history.length; i++) {
      const value = this.history[i]!;
      verifiedEnergy += value * value; verifiedError += (value - predicted(match.start + i)) ** 2;
    }
    if (this.tonal(snapshot, match) && (short || verifiedError > verifiedEnergy * 0.1 ** 2)) {
      return { pcm, echoOnly: false, echoRelated: true };
    }
    const output = Buffer.allocUnsafe(pcm.length);
    let inputEnergy = 0, residualEnergy = 0;
    for (let first = 0; first < count; first += 320) {
      const last = Math.min(count, first + 320);
      let actual = 0, expected = 0, product = 0, error = 0;
      for (let i = first; i < last; i++) {
        const value = pcm.readInt16LE(i * 2), prediction = predicted(start + i);
        actual += value * value; expected += prediction * prediction; product += value * prediction;
        error += (value - prediction) ** 2;
      }
      const coherent = product > 0 && product * product >= actual * expected * 0.68 ** 2 && error < actual * 0.8;
      for (let i = first; i < last; i++) {
        const value = pcm.readInt16LE(i * 2);
        const residual = coherent ? Math.max(-32768, Math.min(32767, Math.round(value - predicted(start + i)))) : value;
        output.writeInt16LE(residual, i * 2);
        inputEnergy += value * value; residualEnergy += residual * residual;
      }
    }
    const tinyResidual = residualEnergy <= Math.max(count * 9, inputEnergy * 0.035 ** 2)
      && verifiedError <= Math.max(9, verifiedEnergy * 0.035 ** 2);
    // Opus re-encoding is not a perfectly linear path. Only suppress its
    // bounded residual when the independent validation block also matches and
    // no substantial independent speech component is detected. Very quiet speech
    // fully masked by the same playback bands remains inherently ambiguous.
    const codecResidual = residualEnergy <= inputEnergy * MAX_CODEC_RESIDUAL ** 2
      && verifiedError <= verifiedEnergy * MAX_HELDOUT_RESIDUAL ** 2 && !hasNovelVoice(output, i => predicted(start + i))
      && coupledResidual(output, snapshot, start);
    const quietResidual = residualEnergy <= count * 24 ** 2 && match.score >= 0.68 ** 2;
    const echoOnly = inputEnergy > count * 16 && (tinyResidual || quietResidual || codecResidual);
    return { pcm: echoOnly ? Buffer.alloc(pcm.length) : output, echoOnly, echoRelated: true };
  }
}
