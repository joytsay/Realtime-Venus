import { t } from "./i18n.js?v=20261007-session-init";

/** Estimate voiced fundamental frequency with a normalized difference function.
 * Unvoiced sound and silence have no reliable pitch and return zero.
 */
export function estimatePitch(samples, sampleRate) {
  const step = Math.max(1, Math.floor(sampleRate / 12000));
  const signal = new Float32Array(Math.floor(samples.length / step));
  let energy = 0;
  for (let i = 0; i < signal.length; i++) {
    for (let j = 0; j < step; j++) signal[i] += samples[i * step + j] / step;
    energy += signal[i] ** 2;
  }
  if (Math.sqrt(energy / signal.length) < 0.003) return 0;
  const rate = sampleRate / step;
  const minLag = Math.floor(rate / 600);
  const maxLag = Math.min(Math.ceil(rate / 70), Math.floor(signal.length / 2));
  const count = signal.length - maxLag;
  const difference = new Float32Array(maxLag + 1);
  let sum = 0;
  difference[0] = 1;
  for (let lag = 1; lag <= maxLag; lag++) {
    let value = 0;
    for (let i = 0; i < count; i++) value += (signal[i] - signal[i + lag]) ** 2;
    sum += value;
    difference[lag] = sum ? (value * lag) / sum : 1;
  }
  for (let lag = Math.max(2, minLag); lag < maxLag; lag++) {
    if (difference[lag] >= 0.15) continue;
    while (lag + 1 < maxLag && difference[lag + 1] < difference[lag]) lag++;
    const left = difference[lag - 1], center = difference[lag], right = difference[lag + 1];
    const curve = left - 2 * center + right;
    const refined = lag + (curve ? Math.max(-0.5, Math.min(0.5, (left - right) / (2 * curve))) : 0);
    const pitch = rate / refined;
    return pitch >= 70 && pitch <= 600 ? pitch : 0;
  }
  return 0;
}

/** Draw actual microphone samples, independently of model speech or network speed. */
export class MicrophoneWaveform {
  constructor(container) {
    this.container = container;
    this.canvas = container.querySelector("canvas");
    this.context = this.canvas.getContext("2d");
    this.label = container.querySelector("#mic-wave-label");
    this.pitchLabel = container.querySelector("#mic-pitch");
    this.icon = container.querySelector("use");
    this.active = false;
    this.muted = false;
    this.pitch = 0;
    this.rms = 0;
    this.lastPitchTime = 0;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvas);
    window.addEventListener("venus-theme", () => this.updateColors());
    this.updateColors();
  }
  updateColors() {
    const style = getComputedStyle(document.documentElement);
    this.color = style.getPropertyValue("--cyan").trim();
    this.baseline = style.getPropertyValue("--line").trim();
  }
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const ratio = Math.min(devicePixelRatio || 1, 2);
    this.width = rect.width;
    this.height = rect.height;
    this.canvas.width = Math.round(rect.width * ratio);
    this.canvas.height = Math.round(rect.height * ratio);
    this.context.setTransform(ratio, 0, 0, ratio, 0, 0);
  }
  setState(active, muted) {
    this.active = active;
    this.muted = muted;
    this.container.hidden = !active;
    this.container.classList.toggle("muted", muted);
    this.icon.setAttribute("href", muted ? "#i-mic-off" : "#i-mic");
    this.label.textContent = t(muted ? "micMuted" : "yourVoice");
    if (!active || muted) {
      this.pitch = 0;
      this.rms = 0;
      this.lastPitchTime = 0;
    }
    this.renderPitch();
  }
  renderPitch() {
    const level = this.active && !this.muted
      ? `${Math.round(20 * Math.log10(Math.max(this.rms, 0.000001)))} dBFS`
      : "";
    this.pitchLabel.textContent = [level, this.pitch ? `${Math.round(this.pitch)} Hz` : ""].filter(Boolean).join(" · ");
  }
  draw(frame, time) {
    if (!this.active || document.hidden) return;
    const ctx = this.context, w = this.width, h = this.height;
    if (!w || !h) return;
    const samples = this.muted ? null : frame?.samples;
    let peak = 0, energy = 0;
    if (samples) {
      for (const value of samples) {
        peak = Math.max(peak, Math.abs(value));
        energy += value * value;
      }
    }
    this.rms = samples?.length ? Math.sqrt(energy / samples.length) : 0;
    const audible = samples && this.rms >= 0.003;
    if (!audible || time - this.lastPitchTime >= 100) {
      this.pitch = audible ? estimatePitch(samples, frame.sampleRate) : 0;
      this.lastPitchTime = time;
      this.renderPitch();
    }
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = this.baseline;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
    // Quiet input still needs visible feedback; pitch requires a stronger signal.
    if (!samples || this.rms < 0.00003) return;

    // A fixed 20 ms window shows pitch as wave spacing and loudness as height.
    // Start at a rising zero crossing to keep sustained notes visually steady.
    const count = Math.min(Math.round(frame.sampleRate * 0.02), samples.length);
    let offset = 0;
    for (let i = 1; i < samples.length - count; i++) {
      if (samples[i - 1] <= 0 && samples[i] > 0) { offset = i; break; }
    }
    const gain = Math.min(80, 3.8 / Math.sqrt(Math.max(0.002, peak)));
    const gradient = ctx.createLinearGradient(0, 0, w, 0);
    gradient.addColorStop(0, this.baseline);
    gradient.addColorStop(0.15, this.color);
    gradient.addColorStop(0.85, this.color);
    gradient.addColorStop(1, this.baseline);
    ctx.strokeStyle = gradient;
    ctx.lineWidth = 1.8;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i < count; i++) {
      const x = (i / (count - 1)) * w;
      const y = h / 2 - Math.tanh(samples[offset + i] * gain) * h * 0.43;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}
