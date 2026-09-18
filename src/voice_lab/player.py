"""Loopback-only dataset sampler and browser player; audio stays on your computer."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import threading
from urllib.parse import urlparse
import uuid
import zipfile

import numpy as np
from scipy import signal
import soundfile as sf

from .dsp import Config
from .recommended import MeasurementConfig, analyze_recommended
from .evaluation import evaluate
from .representation import save_packet, load_packet, decode_packet, MODES
from .datasets import index_dataset, read_registry, write_registry

WEB = Path(__file__).parent / "web"
DESCRIPTIONS = {"synthetic": "Known voiced, unvoiced and silent intervals"}


class Catalog:
    def __init__(self, extra=(), registry=None):
        self.lock = threading.RLock()
        self.registry = Path(registry) if registry is not None else None
        self.registrations = read_registry(self.registry)
        self.import_notes = {}
        self.sources, self.counts = {}, {}
        for name, root in extra:
            if name in self.sources or name == "synthetic":
                raise ValueError(f"Duplicate/reserved dataset name: {name}")
            candidates, _ = index_dataset("folder", root, recursive=True)
            self.sources[name], self.counts[name] = candidates, len(candidates)
        self.sources["synthetic"], self.counts["synthetic"] = [None], 1
        for identity, record in self.registrations.items():
            try:
                paths, skipped = index_dataset(record["kind"], record["path"], record["recursive"])
                self.sources[identity], self.counts[identity] = paths, len(paths)
                self.import_notes[identity] = {"skipped": skipped}
            except (ValueError, OSError) as exc:
                self.sources[identity], self.counts[identity] = [], 0
                self.import_notes[identity] = {"unavailable": str(exc)}

    def add(self, kind, path, name="", recursive=True):
        if not isinstance(path, str) or not path.strip():
            raise ValueError("Choose a folder or CSV path")
        if not isinstance(name, str) or len(name.strip()) > 120:
            raise ValueError("Dataset name must be at most 120 characters")
        if not isinstance(recursive, bool):
            raise ValueError("Recursive scan must be true or false")
        source = Path(path.strip()).expanduser().resolve()
        try:
            paths, skipped = index_dataset(kind, source, recursive)
        except OSError as exc:
            raise ValueError(f"Cannot read dataset source: {exc.strerror}") from exc
        with self.lock:
            for record in self.registrations.values():
                if (record["kind"], record["path"], record["recursive"]) == (kind, str(source), recursive):
                    raise ValueError("This dataset is already added")
            identity = f"local-{uuid.uuid4().hex[:12]}"
            record = {"id": identity, "name": name.strip() or (source.stem if kind == "csv" else source.name),
                      "kind": kind, "path": str(source), "recursive": recursive}
            records = {**self.registrations, identity: record}
            write_registry(self.registry, records)
            self.registrations = records
            self.sources[identity], self.counts[identity] = paths, len(paths)
            self.import_notes[identity] = {"skipped": skipped}
            return next(d for d in self.public() if d["id"] == identity)

    def public(self):
        with self.lock:
            return [{"id": name, "name": self.registrations.get(name, {}).get("name", name),
                     "count": self.counts[name], "indexed": len(items), "available": bool(items),
                     "description": DESCRIPTIONS.get(name, "Local audio recordings"),
                     **({"import": {**self.registrations[name], **self.import_notes[name]}}
                        if name in self.registrations else {})}
                    for name, items in self.sources.items()]

    def sample(self, dataset, seed, seconds, offset=None):
        with self.lock:
            if dataset not in self.sources:
                raise ValueError("Unknown dataset")
            sources = self.sources[dataset]
            label = self.registrations.get(dataset, {}).get("name", dataset)
        if not sources:
            raise ValueError(f"Dataset unavailable: {label}. Check its source folder or CSV.")
        rng = random.Random(seed)
        if dataset == "synthetic":
            audio, reference = synthetic(seed)
            total = len(audio) / 16000
            start = min(max(0, offset or 0), max(0, total - seconds))
            audio = audio[round(start * 16000):round((start + seconds) * 16000)]
            reference = [{"start": max(r["start"], start) - start,
                          "end": min(r["end"], start + len(audio) / 16000) - start,
                          **{k:v for k,v in r.items() if k not in {"start", "end"}},
                          "modulation_offset": max(start-r["start"], 0)} for r in reference
                         if r["end"] > start and r["start"] < start + len(audio) / 16000]
            return audio, 16000, {"dataset": dataset, "recording": "Controlled source-filter signal",
                                 "offset": start, "seed": seed, "total_duration": total,
                                 "reference": reference}
        last_error = None
        # A corrupt/non-audio entry should not break the entire dataset sampler.
        for attempt in range(min(12, len(sources))):
            source = rng.choice(sources)
            try:
                reader = sf.SoundFile(source)
                name = source.name
                identity = str(source)
                with reader:
                    total = len(reader) / reader.samplerate
                    if total < 0.1:
                        continue
                    start = min(max(0, offset), max(0, total - seconds)) if offset is not None else rng.uniform(0, max(0, total - seconds))
                    reader.seek(round(start * reader.samplerate))
                    audio = reader.read(round(seconds * reader.samplerate), dtype="float32", always_2d=True)
                    sr = reader.samplerate
                    codec = reader.subtype
                meta = {"dataset": dataset, "recording": name, "offset": round(start, 4),
                        "dataset_name": label,
                        "seed": seed, "total_duration": round(total, 3),
                        "source_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
                        "reference": None, "codec_subtype": codec}
                return audio, sr, meta
            except (OSError, RuntimeError, ValueError) as exc:
                last_error = exc
                # libsndfile may lack a codec (notably m4a). Decode only the excerpt.
                if shutil.which("ffmpeg"):
                    try:
                        start = max(0, offset or 0)
                        proc = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start),
                                               "-i", str(source), "-t", str(seconds), "-ac", "1",
                                               "-ar", "16000", "-f", "f32le", "pipe:1"],
                                              capture_output=True, timeout=30, check=True)
                        audio = np.frombuffer(proc.stdout, dtype="<f4")
                        if len(audio) >= 1600:
                            return audio, 16000, {"dataset": dataset, "recording": source.name,
                                                 "dataset_name": label,
                                                 "offset": start, "seed": seed, "total_duration": None,
                                                 "reference": None, "codec_fallback": "ffmpeg"}
                    except (subprocess.SubprocessError, OSError):
                        pass
        raise ValueError(f"Could not decode a sample from {dataset}: {last_error}")


def synthetic(seed):
    """Source-filter vowels and fricative-like noise, not a natural-speech benchmark."""
    sr = 16000
    rng = np.random.default_rng(seed)
    x = np.zeros(sr * 7)
    reference = [{"start": 0.6, "end": 2.1, "label": "voiced", "f0_hz": 125,
                  "modulation_hz": 4, "modulation_rate_hz": 2, "formants_hz": [600,1500,2500]},
                 {"start": 2.1, "end": 2.5, "label": "unvoiced"},
                 {"start": 3.3, "end": 5.5, "label": "voiced", "f0_hz": 205,
                  "modulation_hz": 4, "modulation_rate_hz": 2, "formants_hz": [600,1500,2500]}]
    for start, end, pitch in [(0.6, 2.1, 125), (3.3, 5.5, 205)]:
        n = round((end - start) * sr)
        t = np.arange(n) / sr
        phase = np.cumsum(pitch + 4 * np.sin(2 * np.pi * 2 * t)) / sr
        voice = sum(np.sin(2 * np.pi * h * phase) / h for h in range(1, 25))
        for frequency, bandwidth in [(600, 100), (1500, 140), (2500, 200)]:
            radius = np.exp(-np.pi * bandwidth / sr)
            voice = signal.lfilter([1 - radius], [1, -2 * radius * np.cos(2 * np.pi * frequency / sr), radius ** 2], voice)
        voice *= 0.22 / max(np.max(abs(voice)), 1e-12)
        voice *= np.minimum(1, np.minimum(t, t[-1] - t) / 0.02)
        x[round(start * sr):round(start * sr) + n] = voice
    noise = signal.sosfilt(signal.butter(3, 2800, "highpass", fs=sr, output="sos"), rng.normal(size=6400))
    x[33600:40000] = noise * 0.055
    return x.astype(np.float32), reference


def stress(audio, sr, mode, seed):
    if mode == "clean":
        return audio, {"mode": mode}
    rng = np.random.default_rng(seed)
    x = audio.mean(axis=1) if audio.ndim == 2 else audio.copy()
    if mode in {"noise20", "noise10", "noise0"}:
        snr = int(mode[5:])
        noise = rng.normal(size=len(x))
        noise *= np.sqrt(np.mean(x ** 2) / (10 ** (snr / 10) * np.mean(noise ** 2) + 1e-15))
        x = x + noise
        meta = {"mode": mode, "snr_db": snr, "snr_definition": "Whole-excerpt power, including silence"}
    elif mode == "reverb":
        length = round(sr * 0.6)
        t = np.arange(length) / sr
        impulse = rng.normal(size=length) * np.exp(-6.91 * t / 0.45)
        impulse[:round(sr * 0.015)] = 0
        impulse *= 0.45 / (np.linalg.norm(impulse) + 1e-12)
        impulse[0] = 1
        x = signal.fftconvolve(x, impulse)[:len(x)]
        meta = {"mode": mode, "synthetic_t60_seconds": 0.45, "tail_truncated": True}
    elif mode == "hum":
        rms = np.sqrt(np.mean(x ** 2))
        t = np.arange(len(x)) / sr
        x = x + rms * (0.5 * np.sin(2 * np.pi * 100 * t) + 0.25 * np.sin(2 * np.pi * 200 * t))
        meta = {"mode": mode, "fundamental_hz": 100}
    else:
        raise ValueError("Unknown stress condition")
    peak = np.max(abs(x))
    scale = min(1, 0.98 / max(peak, 1e-12))
    return (x * scale).astype(np.float32), {**meta, "common_gain": scale}


def save_result(destination, data, waveforms, packet=None):
    destination.mkdir(parents=True, exist_ok=True)
    for name, audio in waveforms.items():
        # Float WAV keeps the gain unchanged and avoids hidden clipping/normalization.
        sf.write(destination / f"{name}.wav", audio, 16000, subtype="FLOAT")
    if packet is not None:
        save_packet(destination / "representation.npz", packet)
    (destination / "analysis.json").write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    with zipfile.ZipFile(destination / "segments.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(destination / "analysis.json", "analysis.json")
        for i, segment in enumerate(data["segments"]):
            clip = waveforms["original"][round(segment["start"] * 16000):round(segment["end"] * 16000)]
            buffer = io.BytesIO()
            sf.write(buffer, clip, 16000, format="WAV", subtype="FLOAT")
            archive.writestr(f"segment-{i + 1:03}.wav", buffer.getvalue())


def publish_result(output,data,waveforms,provenance,condition,packet=None):
    artifact=uuid.uuid4().hex
    data.update(source=provenance,stress=condition,artifact=artifact)
    if provenance.get("reference"):
        data["evaluation"]=evaluate(data,{"kind":"controlled synthetic; not natural speech", "intervals":provenance["reference"]})
    data["audio"]={name:f"/artifacts/{artifact}/{name}.wav" for name in waveforms}
    data["downloads"]={"json":f"/artifacts/{artifact}/analysis.json","segments":f"/artifacts/{artifact}/segments.zip"}
    if packet is not None:
        data["downloads"]["representation"]=f"/artifacts/{artifact}/representation.npz"
    data["waveform"]=[float(np.max(abs(chunk))) for chunk in np.array_split(waveforms["original"],min(1000,len(waveforms["original"])))]
    save_result(output/artifact,data,waveforms,packet)
    return data


def quality_exclusion(source,condition):
    if condition.get("mode")=="reverb":
        return "synthetic reverberation: pulse/source measurements withheld"
    if source.get("codec_fallback") or source.get("codec_subtype", "").startswith(("MPEG", "VORBIS", "OPUS", "G721", "G723")):
        return "lossy/unknown codec path: pulse/source measurements withheld"
    return None


class PlayerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, catalog, output, picker=None):
        self.catalog, self.output = catalog, output
        self.picker = picker
        self.slots = threading.BoundedSemaphore(2)
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def _same_host(self):
        expected = f"127.0.0.1:{self.server.server_port}"
        valid_hosts = {expected, f"localhost:{self.server.server_port}"}
        if self.headers.get("Host") not in valid_hosts:
            self.send_error(403, "Loopback host required")
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{h}" for h in valid_hosts}:
            self.send_error(403, "Same-origin requests required")
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self.send_error(403, "Same-origin requests required")
            return False
        return True

    def _json(self, value, status=200):
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_POST(self):
        if not self._same_host():
            return
        if self.path not in {"/api/sample", "/api/datasets", "/api/dataset-picker"}:
            self.send_error(404)
            return
        acquired = False
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 8192:
                raise ValueError("JSON request size must be 1–8192 bytes")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("Send a JSON object")
            if self.path == "/api/dataset-picker":
                if self.server.picker is None:
                    self._json({"error": "Native Browse is available in the Mac app; paste a local path here"}, 400)
                else:
                    kind = request.get("kind")
                    if kind not in {"folder", "csv"}:
                        raise ValueError("Choose folder or csv")
                    self._json({"path": self.server.picker(kind)})
                return
            if self.path == "/api/datasets":
                dataset = self.server.catalog.add(request.get("kind"), request.get("path"),
                                                  request.get("name", ""), request.get("recursive", True))
                self._json({"dataset": dataset, "datasets": self.server.catalog.public()}, 201)
                return
            seed = int(request.get("seed", 42))
            seconds = float(request.get("seconds", 20))
            offset = request.get("offset")
            if not math.isfinite(seconds) or not 1 <= seconds <= 45:
                raise ValueError("Excerpt must be between 1 and 45 seconds")
            if offset is not None:
                offset = float(offset)
                if not math.isfinite(offset) or offset < 0:
                    raise ValueError("Offset must be a finite nonnegative number")
            sensitivity = float(request.get("sensitivity", 0.5))
            f0_min, f0_max = float(request.get("f0_min", 60)), float(request.get("f0_max", 500))
            cfg = replace(Config(), sensitivity=sensitivity, denoise=bool(request.get("denoise", False)),
                          f0_min=f0_min, f0_max=f0_max, pitch_window_ms=max(50, 3000 / f0_min) if f0_min > 0 else 50)
            cfg.validate()
            if request.get("method", "recommended") != "recommended":
                raise ValueError("The previous segmenter and method comparison were removed; use the recommended pipeline")
            mc=replace(MeasurementConfig(),channel=request.get("channel","broadband"),
                       formant_ceiling=float(request.get("formant_ceiling",5500)),
                       noise_method=request.get("noise_method","wiener"))
            mc.validate()
            acquired = self.server.slots.acquire(blocking=False)
            if not acquired:
                self._json({"error": "Two analyses are already running; try again shortly"}, 429)
                return
            audio, sr, provenance = self.server.catalog.sample(request["dataset"], seed, seconds, offset)
            if audio.ndim==2:
                selected=int(np.argmax(np.mean(audio**2,axis=0)))
                provenance.update(input_channels=audio.shape[1],selected_channel=selected)
                audio=audio[:,selected]
            audio, condition = stress(audio, sr, request.get("stress", "clean"), seed)
            exclusion=quality_exclusion(provenance,condition)
            data,waves,packet=analyze_recommended(audio,sr,cfg,mc,quality_exclusion=exclusion,return_packet=True)
            packet["metadata"]["source"]=provenance
            packet["metadata"]["stress"]=condition
            data=publish_result(self.server.output,data,waves,provenance,condition,packet)
            self._json(data)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:
            print(f"Analysis failed: {exc}", flush=True)
            self._json({"error": "Analysis failed; see the local server log"}, 500)
        finally:
            if acquired:
                self.server.slots.release()

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_GET(self, head=False):
        if not self._same_host():
            return
        path = urlparse(self.path).path
        if path == "/api/datasets":
            self._json({"datasets": self.server.catalog.public(), "native_picker": self.server.picker is not None})
            return
        public = {"/": (WEB / "index.html", "text/html"),
                  "/app.js": (WEB / "app.js", "text/javascript"),
                  "/style.css": (WEB / "style.css", "text/css")}
        if path in public:
            file, content_type = public[path]
        elif path.startswith("/artifacts/"):
            parts = path.split("/")
            allowed = {"original.wav", "speech.wav", "enhanced.wav", "minimal.wav", "glottal.wav", "glottal_derivative.wav",
                       "analysis.json", "segments.zip", "representation.npz"} | {f"packet_{mode}.wav" for mode in MODES}
            if len(parts) != 4 or len(parts[2]) != 32 or any(c not in "0123456789abcdef" for c in parts[2]) or parts[3] not in allowed:
                self.send_error(404)
                return
            file = self.server.output / parts[2] / parts[3]
            content_type = "audio/wav" if file.suffix == ".wav" else "application/json" if file.suffix == ".json" else "application/zip" if file.suffix == ".zip" else "application/octet-stream"
        else:
            self.send_error(404)
            return
        if not file.is_file():
            self.send_error(404)
            return
        size = file.stat().st_size
        start, end, partial = 0, size - 1, False
        byte_range = self.headers.get("Range")
        if byte_range:
            try:
                unit, values = byte_range.split("=", 1)
                first, last = values.split("-", 1)
                if unit != "bytes" or "," in values:
                    raise ValueError()
                if not first:
                    suffix = int(last)
                    if suffix <= 0:
                        raise ValueError()
                    start = max(0, size - suffix)
                else:
                    start = int(first)
                    end = min(int(last), size - 1) if last else size - 1
                if not 0 <= start <= end < size:
                    raise ValueError()
                partial = True
            except (ValueError, TypeError):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not head:
            with file.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining:
                    chunk = stream.read(min(65536, remaining))
                    self.wfile.write(chunk)
                    remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--dataset", action="append", default=[], metavar="NAME=PATH")
    serve.add_argument("--out", type=Path, default=Path.cwd() / "voice-lab-runs")
    serve.add_argument("--registry", type=Path, help="Remember datasets added through the UI; defaults to OUT/datasets.json")
    process = commands.add_parser("analyze")
    process.add_argument("audio", type=Path)
    process.add_argument("--out", type=Path, required=True)
    process.add_argument("--offset", type=float, default=0)
    process.add_argument("--seconds", type=float, default=30)
    process.add_argument("--denoise", action="store_true")
    process.add_argument("--channel", choices=["broadband","telephone","noisy"],default="broadband")
    process.add_argument("--formant-ceiling", type=float, default=5500)
    process.add_argument("--reference", type=Path,help="JSON with excerpt-relative intervals and optional reference tracks")
    decoder = commands.add_parser("decode", help="Synthesize from a six-stream packet alone")
    decoder.add_argument("packet", type=Path)
    decoder.add_argument("--out", type=Path, required=True)
    decoder.add_argument("--mode", choices=MODES, default="full")
    args = parser.parse_args()
    if args.command == "decode":
        packet=load_packet(args.packet)
        wave=decode_packet(packet,args.mode)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        sf.write(args.out,wave,packet["metadata"]["sample_rate"],subtype="FLOAT")
        print(json.dumps({"schema":packet["metadata"]["schema"],"mode":args.mode,"samples":len(wave),"output":str(args.out.resolve())}))
        return
    if args.command == "analyze":
        if not math.isfinite(args.seconds) or not 0 < args.seconds <= 600 or not math.isfinite(args.offset) or args.offset < 0:
            parser.error("Use a nonnegative offset and 0 < seconds <= 600")
        with sf.SoundFile(args.audio) as stream:
            stream.seek(round(args.offset * stream.samplerate))
            audio, sr = stream.read(round(args.seconds * stream.samplerate), always_2d=True), stream.samplerate
            codec=stream.subtype
        source={"file":str(args.audio.resolve()),"offset":args.offset,"codec_subtype":codec}
        if audio.ndim==2:
            selected=int(np.argmax(np.mean(audio**2,axis=0)))
            source.update(input_channels=audio.shape[1],selected_channel=selected)
            audio=audio[:,selected]
        cfg=replace(Config(),denoise=args.denoise)
        data,waves,packet=analyze_recommended(audio,sr,cfg,replace(MeasurementConfig(),channel=args.channel,formant_ceiling=args.formant_ceiling),quality_exclusion=quality_exclusion(source,{}),return_packet=True)
        packet["metadata"]["source"]=source
        data.update(source=source)
        if args.reference:
            data["evaluation"]=evaluate(data,json.loads(args.reference.read_text()))
        save_result(args.out,data,waves,packet)
        print(json.dumps({"stats":data["stats"],"representation":{k:data["representation"][k] for k in ("schema","full_rmse","full_max_error")}},indent=2))
        return
    extra = []
    for value in args.dataset:
        name, separator, path = value.partition("=")
        if not separator or not name or not Path(path).is_dir():
            parser.error("--dataset requires NAME=PATH to an existing directory")
        extra.append((name, Path(path)))
    print("Indexing local datasets…", flush=True)
    catalog = Catalog(extra=extra, registry=args.registry or args.out / "datasets.json")
    args.out.mkdir(parents=True, exist_ok=True)
    server = PlayerServer(("127.0.0.1", args.port), catalog, args.out)
    print(f"Voice segmentation player: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Available: {', '.join(catalog.sources)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
