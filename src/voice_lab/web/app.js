"use strict";
// Opening the source HTML directly should lead to the running local player.
if (window.location.protocol === "file:") {
  window.location.replace("http://127.0.0.1:8765/");
} else {
const $ = id => document.getElementById(id);
const audio = $("audio");
let analysis = null, datasets = [], track = "original", segmentEnd = null, busy = false;
let searchId = null, pinnedSample = null;
const colors = {voiced: "#43d8ce", unvoiced: "#729ffc", nonspeech: "#586879"};
const notes = {
  original: "Original recording, resampled to 16 kHz mono. Stress conditions are applied before analysis.",
  speech: "Speech intervals on the original timeline, with short fades. Detected nonspeech is muted.",
  enhanced: "Estimator audio with channel-aware band limiting and optional conservative noise reduction using detected nonspeech.",
  minimal: "DC removal and gentle zero-phase highpass. Pulse quality, intensity and IAIF use this branch, without denoising.",
  glottal: "Experimental IAIF source estimate in qualified intervals only; zero elsewhere. Relative amplitude is uncalibrated."
};
const packetLabels = {full:"Full packet",harmonic_only:"Harmonics only",residual_only:"Residual only",parametric:"Parametric · no stored residual",flat_tract:"Flat tract",constant_gain:"Constant gain"};
const outputLabels = {original:"Original",speech:"Speech only",minimal:"Minimal branch",enhanced:"Filtered / enhanced",glottal:"IAIF source estimate",...Object.fromEntries(Object.entries(packetLabels).map(([key,label])=>[`packet_${key}`,label]))};
Object.assign(notes, {
  packet_full:"Decoded from the six-stream packet alone. Target: minimal branch. Harmonics + excitation remainder → LSF tract → gain; all intervals retained.",
  packet_harmonic_only:"Modeled voiced harmonics through the tract, with original excitation gain. The complementary residual is omitted.",
  packet_residual_only:"Excitation remainder through the tract, with original gain. Includes unvoiced detail, transients and any periodic detail the harmonic model missed.",
  packet_parametric:"Stored harmonic amplitudes/phases + deterministic band-shaped noise through the tract, scaled to measured frame loudness. Detailed residual samples are omitted; waveform fidelity is approximate.",
  packet_flat_tract:"Full excitation and gain with the tract filter bypassed. This exposes the modeled source rather than intelligible reconstructed voice.",
  packet_constant_gain:"Full excitation and tract with a constant median output RMS in nonzero frames. Silence remains zero; this tests the amplitude stream."
});
function time(value) { const m = Math.floor(value / 60); return `${m}:${(value % 60).toFixed(2).padStart(5, "0")}`; }
function context(id) {
  const canvas = $(id), rect = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  // Keep CSS height fixed; resizing backing pixels must not grow the chart.
  const logical = id === "waveform" ? 180 : id === "spectrum" ? 220 : 160;
  canvas.style.height = `${logical}px`;
  canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(logical * ratio);
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  return {ctx, w:rect.width, h:logical};
}
function drawWave() {
  const {ctx,w,h} = context("waveform");
  ctx.strokeStyle = "#263444"; ctx.beginPath(); ctx.moveTo(0,h/2); ctx.lineTo(w,h/2); ctx.stroke();
  if (!analysis) return;
  for (const r of analysis.regions) {
    const left = r.start / analysis.duration * w, width = (r.end-r.start) / analysis.duration * w;
    ctx.globalAlpha = r.label === "nonspeech" ? 0.08 : 0.12; ctx.fillStyle=colors[r.label]; ctx.fillRect(left,0,width,h-15);
    ctx.globalAlpha=1; ctx.fillRect(left,h-8,width,4);
  }
  const peak = Math.max(...analysis.waveform, 0.001);
  ctx.strokeStyle = "#aec5d5"; ctx.lineWidth = 1;
  analysis.waveform.forEach((value,i) => { const x=i/analysis.waveform.length*w, amplitude=value/peak*(h-30)/2;
    ctx.beginPath(); ctx.moveTo(x,h/2-amplitude); ctx.lineTo(x,h/2+amplitude); ctx.stroke(); });
  for (const boundary of analysis.speaker_changes) {
    const x=boundary.time/analysis.duration*w; ctx.strokeStyle="#f2b56b"; ctx.setLineDash([3,4]);
    ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();ctx.setLineDash([]);
  }
  if (analysis.source.reference) for (const r of analysis.source.reference) {
    ctx.fillStyle="#f2b56b";ctx.fillRect(r.start/analysis.duration*w,h-15,(r.end-r.start)/analysis.duration*w,2);
  }
  const x=audio.currentTime/analysis.duration*w;ctx.strokeStyle="#ffffff";ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();
}
function drawTrace(id, series, ceiling, palette) {
  const {ctx,w,h}=context(id), left=42, top=12, bottom=h-20, width=w-left-6;
  ctx.font="12px system-ui";ctx.fillStyle="#9aadc0";
  for (let j=0;j<=3;j++) { const value=ceiling*j/3, y=bottom-value/ceiling*(bottom-top);
    ctx.fillText(String(Math.round(value)),0,y+4);ctx.strokeStyle="#263444";ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w,y);ctx.stroke(); }
  if (!analysis) return;
  series.forEach((values,k) => {ctx.strokeStyle=palette[k];ctx.lineWidth=1.6;ctx.beginPath();let started=false;
    values.forEach((value,i)=>{if(value===null||value<=0||value>ceiling){started=false;return;}const x=left+analysis.frames.time[i]/analysis.duration*width,y=bottom-value/ceiling*(bottom-top);
      if(started)ctx.lineTo(x,y);else ctx.moveTo(x,y);started=true;});ctx.stroke();});
}
function drawAll(){
  drawWave();
  if(!$("full-features").open || !analysis)return;
  const candidates = analysis?.frames.pitch_estimators && $("pitch-candidates").checked;
  const pitchSeries = analysis ? candidates ? [0,1,2,3].map(k=>analysis.frames.pitch_estimators.map(row=>row[k])).concat([analysis.frames.f0]) : [analysis.frames.f0] : [];
  drawTrace("pitch",pitchSeries,analysis?analysis.config.f0_max:500,candidates?["#729ffc","#d0a3ff","#f2b56b","#70818f",colors.voiced]:[colors.voiced]);
  drawTrace("formants",analysis?[0,1,2].map(k=>analysis.frames.formants.map(row=>row[k])):[],analysis?.formant_analysis?.ceiling_hz||5000,[colors.voiced,colors.unvoiced,"#f2b56b"]);
  drawSpectrum();drawIntensity();if(hasSource(analysis))drawSource();drawPacket();
}
function drawSigned(id,values,minimum,maximum,times){
  const {ctx,w,h}=context(id),left=42,width=w-left-6,top=12,bottom=h-20;
  ctx.font="11px system-ui";ctx.fillStyle="#9aadc0";
  for(let j=0;j<=4;j++){const v=minimum+(maximum-minimum)*j/4,y=bottom-j/4*(bottom-top);ctx.fillText(String(Math.round(v)),0,y+4);ctx.strokeStyle="#263444";ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w,y);ctx.stroke();}
  ctx.strokeStyle=colors.voiced;ctx.beginPath();let started=false;
  values.forEach((v,i)=>{if(v===null){started=false;return;}const x=left+times[i]/analysis.duration*width,y=bottom-(Math.max(minimum,Math.min(maximum,v))-minimum)/(maximum-minimum)*(bottom-top);if(started)ctx.lineTo(x,y);else ctx.moveTo(x,y);started=true;});ctx.stroke();
}
function drawPacket(){
  const r=analysis?.representation;if(!r)return;
  const d=r.display,order=d.lsf_hz[0].length;
  drawTrace("packet-tract",Array.from({length:order},(_,k)=>d.lsf_hz.map(row=>row[k])),8000,Array.from({length:order},(_,k)=>`hsl(${170+k*9} 60% 65%)`));
  drawSigned("packet-gain",d.gain_db,-100,0,d.time);drawSigned("packet-hnr",d.hnr_db,-50,50,d.time);
  const {ctx,w,h}=context("packet-aperiodicity"),edges=r.metadata.band_edges_hz,left=42,width=w-left;
  for(let i=0;i<d.aperiodicity.length;i++)for(let k=0;k<edges.length-1;k++){
    const v=d.gain_db[i]<-100?0:d.aperiodicity[i][k];ctx.fillStyle=`rgb(${Math.round(12+210*v)},${Math.round(25+155*v)},${Math.round(40+60*v)})`;
    ctx.fillRect(left+i/d.aperiodicity.length*width,(edges.length-2-k)/(edges.length-1)*(h-20),width/d.aperiodicity.length+1,(h-20)/(edges.length-1)+1);
  }
  ctx.fillStyle="#9aadc0";ctx.font="11px system-ui";for(let k=0;k<edges.length-1;k++)ctx.fillText(String(edges[k]),0,(edges.length-2-k)/(edges.length-1)*(h-20)+12);
}
function drawSpectrum(){
  const {ctx,w,h}=context("spectrum"),type=$("spectral-view").value,s=analysis.spectral;
  const values=type==="stft"?s.stft_db:type==="cqt"?s.cqt_db:s.mfcc;
  const bins=values[0].length,left=40,width=w-left;
  let maximum=-Infinity,minimum=Infinity;
  for(const row of values)for(const v of row){maximum=Math.max(maximum,v);minimum=Math.min(minimum,v);}
  if(type!=="mfcc")minimum=Math.max(minimum,maximum-65);
  const columns=Math.max(1,Math.min(Math.ceil(width),values.length));
  for(let column=0;column<columns;column++){
    const row=values[Math.floor(column/columns*values.length)];
    for(let j=0;j<bins;j++){
      const strength=Math.max(0,Math.min(1,(row[j]-minimum)/(maximum-minimum||1)));
      ctx.fillStyle=`rgb(${Math.round(12+220*strength)},${Math.round(22+185*strength)},${Math.round(35+95*strength)})`;
      ctx.fillRect(left+column/columns*width,(bins-1-j)/bins*(h-20),width/columns+1,(h-20)/bins+1);
    }
  }
  ctx.fillStyle="#9aadc0";ctx.font="11px system-ui";
  const freq=type==="stft"?s.stft_frequency_hz:type==="cqt"?s.cqt_frequency_hz:null;
  for(let j=0;j<=3;j++){const k=Math.min(bins-1,Math.floor(j*(bins-1)/3));ctx.fillText(freq?String(Math.round(freq[k])):String(k),0,(bins-1-k)/bins*(h-20)+10);}
  ctx.fillText("0 s",left,h-3);ctx.fillText(`${analysis.duration.toFixed(1)} s`,Math.max(left,w-42),h-3);
}
function drawIntensity(){
  const {ctx,w,h}=context("intensity"),values=analysis.frames.intensity_dbfs,left=40,width=w-left;
  ctx.font="11px system-ui";ctx.fillStyle="#9aadc0";
  for(const db of [-80,-60,-40,-20,0]){const y=(0-db)/80*(h-20);ctx.fillText(String(db),0,y+10);}
  ctx.strokeStyle=colors.voiced;ctx.beginPath();values.forEach((v,i)=>{const x=left+analysis.frames.time[i]/analysis.duration*width,y=Math.min(h-20,Math.max(0,-v/80*(h-20)));if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();
  for(const event of analysis.landmarks){ctx.fillStyle="#f2b56b";ctx.fillRect(left+event.time/analysis.duration*width,h-15,2,8);}
  for(const nucleus of analysis.nuclei){ctx.fillStyle=colors.unvoiced;ctx.beginPath();ctx.arc(left+nucleus.time/analysis.duration*width,h-20,3,0,Math.PI*2);ctx.fill();}
}
function drawSource(){
  const {ctx,w,h}=context("source-flow"),series=analysis.glottal.display;
  if(!series)return;
  for(const [k,values]of[series.flow,series.derivative].entries()){
    const peak=Math.max(...values.map(Math.abs),1e-12);ctx.strokeStyle=k?"#f2b56b":colors.voiced;ctx.beginPath();
    values.forEach((v,i)=>{const x=i/values.length*w,y=h/2-v/peak*(h/2-10);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();
  }
}
function hasSource(data){return data.glottal.frames.length>0;}
function renderOutputs(data){
  $("output-panel").hidden=false;
  const body=$("outputs").querySelector("tbody");body.replaceChildren();
  for(const [name,label]of Object.entries(outputLabels)){
    if(!data.audio[name] || (name==="glottal"&&!hasSource(data)))continue;
    const row=document.createElement("tr"),cell=document.createElement("td"),button=document.createElement("button");
    button.dataset.track=name;button.textContent=label;cell.append(button);row.append(cell);
    const metrics=data.representation.modes[name.replace(/^packet_/,"")];
    for(const value of[metrics?metrics.rmse_to_minimal.toExponential(2):"—",metrics?metrics.peak.toFixed(3):"—"]){const cell=document.createElement("td");cell.textContent=value;row.append(cell);}
    body.append(row);
  }
  markTrack();
}
function markTrack(){
  $("outputs").querySelectorAll("[data-track]").forEach(button=>{
    const selected=button.dataset.track===track;button.setAttribute("aria-pressed",String(selected));button.closest("tr").classList.toggle("selected",selected);
  });
}
function renderFeatures(data){
  $("full-features").hidden=false;
  $("formant-title").textContent="Burg + Kalman / RTS formants";
  $("evaluation-note").textContent=data.evaluation ? `Controlled reference · activity F1 ${data.evaluation.activity.f1?.toFixed(3)||"—"} · pitch gross error ${data.evaluation.pitch?.gross_error_rate!=null?`${(data.evaluation.pitch.gross_error_rate*100).toFixed(1)}%`:"—"}. These metrics use synthetic annotations.` : "No reference annotations loaded for this recording.";
  const r=data.representation;
  $("packet-note").textContent=`${r.frame_rate_hz.toFixed(0)} frames/s · target: minimal branch · full packet RMS error ${r.full_rmse.toExponential(2)} · ${(r.array_bytes/1024/1024).toFixed(1)} MB of unquantized arrays. Dense residual samples preserve detail; this is not a compact codec or speaker separator.`;
  $("download-packet").href=data.downloads.representation;$("download-packet").hidden=!data.downloads.representation;
  $("stream-list").replaceChildren();for(const [key,value]of Object.entries(r.streams)){const p=document.createElement("p");p.className="muted";p.textContent=`${key}: ${value}`;$("stream-list").append(p);}
  $("timing-note").textContent=`${data.nuclei.length} estimated vowel-like nuclei · ${data.stats.nuclei_per_elapsed_second.toFixed(2)} / elapsed second · ${data.stats.nuclei_per_speech_second?.toFixed(2)||"—"} / speech second. Amber: landmarks. Blue: nuclei.`;
  $("landmarks").replaceChildren();for(const e of data.landmarks){const b=document.createElement("button");b.textContent=`${e.label} · ${time(e.time)}`;b.onclick=()=>playInterval(Math.max(0,e.time-.15),Math.min(data.duration,(e.end||e.time)+.2));$("landmarks").append(b);}
  const sourceAvailable=hasSource(data);
  $("source-flow").hidden=!sourceAvailable;
  $("quality").replaceChildren();for(const r of data.voice_quality){
    if(!sourceAvailable&&r.status==="rejected")continue;
    const p=document.createElement("p");p.className="muted";p.textContent=r.status==="rejected"?`Withheld${r.start!=null?` ${time(r.start)}–${time(r.end)}`:""}: ${r.reason}`:`${time(r.start)}–${time(r.end)} · local jitter ${(r.jitter_local*100).toFixed(3)}% · local shimmer ${(r.shimmer_local*100).toFixed(3)}% · shimmer ${r.shimmer_db.toFixed(3)} dB · ${r.accepted_period_pairs} accepted cycle pairs`;$("quality").append(p);
  }
  const naq=data.glottal.intervals.filter(r=>r.naq!=null).map(r=>`${time(r.start)}: NAQ proxy ${r.naq.toFixed(3)}`).join(" · ");
  const reason=data.voice_quality.find(r=>r.status==="rejected")?.reason||data.glottal.reason;
  $("source-note").textContent=sourceAvailable?`IAIF frames: ${data.glottal.frames.length}. ${naq} · teal flow / amber derivative peak envelopes, separately scaled. Experimental source estimates, not calibrated physiology.`:`IAIF withheld: ${reason}.`;
  $("feature-record").textContent=JSON.stringify({parameters:data.measurement_config,provenance:data.provenance,representation:{metadata:r.metadata,identity_fallback_frames:r.identity_fallback_frames,full_max_error:r.full_max_error},coverage:{pitch:data.stats.pitch_coverage,formants:data.stats.formant_coverage},evaluation:data.evaluation||"No reference annotations loaded"},null,2);
}
function setTrack(name){
  if(!analysis?.audio[name] || (name==="glottal"&&!hasSource(analysis)) || name===track)return;
  track=name;segmentEnd=null;
  if(!busy&&name.startsWith("packet_"))$("error-search-mode").value=name.replace(/^packet_/,"");
  markTrack();
  $("track-note").textContent=trackNote(name);
  const position=audio.currentTime, playing=!audio.paused;
  audio.src=analysis.audio[name];
  audio.onloadedmetadata=()=>{audio.currentTime=Math.min(position,analysis.duration);if(playing)audio.play().catch(showPlaybackError);audio.onloadedmetadata=null;};
}
function trackNote(name){
  if(name==="enhanced"&&analysis)return `Estimator branch, bandlimited to ${analysis.provenance.bandwidth_hz} Hz; ${analysis.provenance.enhancement.applied?analysis.provenance.enhancement.method:"no denoising"}. Physical pulse measurements use the minimal branch.`;
  return notes[name];
}
function showPlaybackError(error){$("error").hidden=false;$("error").textContent=`Playback could not start: ${error.message}`;}
function playInterval(start,end){segmentEnd=end;audio.currentTime=start;audio.play().catch(showPlaybackError);}
function render(data){
  audio.pause();segmentEnd=null;analysis=data;
  if(!data.audio[track] || (track==="glottal"&&!hasSource(data)))track="original";
  renderOutputs(data);
  audio.onloadedmetadata=null;
  $("recording-title").textContent=data.source.recording;
  $("provenance").textContent=`${data.source.dataset_name||data.source.dataset} · seed ${data.source.seed} · source offset ${time(data.source.offset)} · ${data.duration.toFixed(2)} s excerpt`;
  $("speech-stat").textContent=`${data.stats.speech_seconds.toFixed(1)} s`;
  $("segment-stat").textContent=data.segments.length;
  $("pitch-stat").textContent=data.stats.median_f0?`${Math.round(data.stats.median_f0)} Hz`:"—";
  $("voiced-stat").textContent=`${Math.round(data.stats.voiced_fraction*100)}%`;
  $("duration-label").textContent=time(data.duration);
  audio.src=data.audio[track];$("track-note").textContent=trackNote(track);
  const list=$("segments");list.replaceChildren();
  if(!data.segments.length){const p=document.createElement("p");p.className="muted";p.textContent="No speech intervals detected. Try adjusting sensitivity or choose another sample.";list.append(p);}
  data.segments.forEach((s,i)=>{const button=document.createElement("button");button.className="segment";
    const number=document.createElement("b");number.textContent=String(i+1).padStart(2,"0");
    const label=document.createElement("span");label.textContent=`${time(s.start)} – ${time(s.end)}`;
    const small=document.createElement("small");small.textContent=`${(s.end-s.start).toFixed(2)} s · ${Math.round(s.voiced_fraction*100)}% voiced · cue score ${s.confidence.toFixed(2)}`;
    label.append(small);button.append(number,label);button.addEventListener("click",()=>playInterval(s.start,s.end));list.append(button);});
  $("changes").replaceChildren();data.speaker_changes.forEach(c=>{const b=document.createElement("button");b.textContent=`Possible speaker change · ${time(c.time)}`;b.onclick=()=>playInterval(Math.max(0,c.time-1),Math.min(data.duration,c.time+1));$("changes").append(b);});
  for(const [id,url]of[["download-json",data.downloads.json],["download-segments",data.downloads.segments]]){$(id).href=url;$(id).hidden=false;}
  $("warnings").replaceChildren();data.warnings.forEach(text=>{const p=document.createElement("p");p.textContent=text;$("warnings").append(p);});
  $("reference-note").textContent=data.source.reference?"Amber markers show known synthetic speech intervals. This controlled source-filter signal is not natural speech. Real recordings have no loaded segmentation ground truth.":"These labels are predictions. No segmentation ground truth is loaded for this recording.";
  $("transcript").textContent=data.source.transcript?`Transcript: ${data.source.transcript}`:"";
  renderFeatures(data);drawAll();
}
function describe(){const d=datasets.find(d=>d.id===$("dataset").value);$("dataset-description").textContent=d?d.import?.unavailable?`Unavailable: ${d.import.unavailable}`:`${d.description} · ${d.count.toLocaleString()} recordings (${d.indexed.toLocaleString()} in sampling pool)`:"";updateSearchScope();}
function updateDatasets(items,selected){
  datasets=items;$("dataset").replaceChildren();
  for(const d of datasets){const option=document.createElement("option");option.value=d.id;option.textContent=(d.id==="synthetic"?"Synthetic · known intervals":d.name||d.id)+(d.available===false?" · unavailable":"");option.disabled=d.available===false;$("dataset").append(option);}
  $("dataset").value=selected||datasets.find(d=>d.available!==false)?.id;describe();
}
function importMode(){
  const csv=$("dataset-kind").value==="csv";
  $("dataset-path-label").textContent=csv?"CSV path":"Folder path";
  $("dataset-path").placeholder=csv?"/Users/you/dataset.csv":"/Users/you/Recordings";
  $("recursive-option").hidden=csv;$("csv-hint").hidden=!csv;
}
function importError(error){$("dataset-error").textContent=error.message;$("dataset-error").hidden=false;}
$("add-dataset").onclick=()=>{$("dataset-form").reset();importMode();$("dataset-error").hidden=true;$("dataset-dialog").showModal();};
$("close-dataset").onclick=$("cancel-dataset").onclick=()=>$("dataset-dialog").close();
$("dataset-kind").onchange=importMode;
$("browse-dataset").onclick=async()=>{
  $("browse-dataset").disabled=true;$("dataset-error").hidden=true;
  try{const response=await fetch("/api/dataset-picker",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({kind:$("dataset-kind").value})}),data=await response.json();if(!response.ok)throw new Error(data.error);if(data.path)$("dataset-path").value=data.path;}catch(error){importError(error);}finally{$("browse-dataset").disabled=false;}
};
$("dataset-form").onsubmit=async event=>{
  event.preventDefault();$("import-dataset").disabled=true;$("import-dataset").textContent="Scanning…";$("dataset-error").hidden=true;
  try{
    const response=await fetch("/api/datasets",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({kind:$("dataset-kind").value,path:$("dataset-path").value,name:$("dataset-name").value,recursive:$("dataset-kind").value==="folder"?$("dataset-recursive").checked:false})}),data=await response.json();
    if(!response.ok)throw new Error(data.error||"Could not add dataset");
    updateDatasets(data.datasets,data.dataset.id);
    const skipped=Object.entries(data.dataset.import.skipped).map(([reason,count])=>`${count} ${reason}`).join(", ");
    $("dataset-result").textContent=`Added ${data.dataset.count.toLocaleString()} audio files.${skipped?` Skipped: ${skipped}.`:""}`;$("dataset-result").hidden=false;
    $("dataset-dialog").close();await sample();
  }catch(error){importError(error);}finally{$("import-dataset").disabled=false;$("import-dataset").textContent="Add dataset";}
};
function updateSearchScope(){
  const d=datasets.find(d=>d.id===$("dataset").value);
  $("error-search-scope").textContent=`${d?`${d.name||d.id}: ${d.indexed.toLocaleString()} indexed of ${d.count.toLocaleString()} discovered recordings. `:""}One excerpt per tested recording, using current settings. Highest among tested excerpts, not every time window. Full packet errors are normally numerical roundoff.`;
}
function sampleRequest(overrides={}){
  const noiseReduction=$("noise-reduction").value;
  const request={dataset:$("dataset").value,seed:Number($("seed").value),seconds:Number($("seconds").value),stress:$("stress").value,sensitivity:Number($("sensitivity").value),f0_min:Number($("f0-min").value),f0_max:Number($("f0-max").value),denoise:noiseReduction!=="off",channel:$("channel").value,formant_ceiling:Number($("formant-ceiling").value),noise_method:noiseReduction==="off"?"wiener":noiseReduction,...overrides};
  if(pinnedSample&&request.dataset===pinnedSample.dataset&&request.seed===pinnedSample.seed)request.source_index=pinnedSample.source_index;
  return request;
}
function setBusy(value){
  busy=value;
  for(const id of["sample","shuffle","add-dataset","dataset","seconds","seed","stress","noise-reduction","sensitivity","f0-min","f0-max","channel","formant-ceiling","error-search-mode","error-search-count","error-search-start"])$(id).disabled=value;
}
async function sample(overrides={}){
  if(busy)return {error:"An analysis is already running"};
  setBusy(true);audio.pause();$("status").textContent="Analyzing…";$("error").hidden=true;$("error-search-result").hidden=true;$("error-search-progress").hidden=true;
  try{
    const response=await fetch("/api/sample",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(sampleRequest(overrides))});
    const data=await response.json();if(!response.ok)throw new Error(data.error||"Analysis failed");
    render(data);$("status").textContent="Analysis ready";return {artifact:data.artifact,segments:data.segments,stats:data.stats};
  }catch(error){$("error").textContent=error.message;$("error").hidden=false;$("status").textContent="Could not analyze";return {error:error.message};}
  finally{setBusy(false);}
}
async function findHighestError(){
  if(busy||!$("sample-form").reportValidity())return;
  const count=$("error-search-count").value;
  const request={...sampleRequest(),mode:$("error-search-mode").value,count:count==="all"?"all":Number(count)};
  delete request.source_index;
  setBusy(true);audio.pause();$("error").hidden=true;$("status").textContent="Searching…";
  $("error-search-result").hidden=false;$("error-search-result").textContent="Starting local search…";
  $("error-search-progress").hidden=false;$("error-search-progress").value=0;
  try{
    const response=await fetch("/api/error-search",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(request)});
    let job=await response.json();if(!response.ok)throw new Error(job.error||"Could not start search");
    searchId=job.id;$("error-search-cancel").hidden=false;$("error-search-cancel").disabled=false;
    for(;;){
      $("error-search-progress").max=job.total;$("error-search-progress").value=job.processed;
      $("error-search-result").textContent=`${packetLabels[job.mode]} · ${job.processed}/${job.total} tested · ${job.failed} skipped${job.best_rmse!=null?` · highest RMS error ${job.best_rmse.toExponential(3)}`:""}`;
      if(job.state!=="running")break;
      await new Promise(resolve=>setTimeout(resolve,700));
      const response=await fetch(`/api/error-search/${searchId}`);job=await response.json();
      if(!response.ok)throw new Error(job.error||"Could not read search progress");
    }
    if(job.result){
      const winner=job.result.error_search.winner_request;
      pinnedSample={dataset:winner.dataset,seed:winner.seed,source_index:winner.source_index};
      $("seed").value=winner.seed;track=`packet_${job.mode}`;render(job.result);
      $("error-search-result").textContent=`${job.state==="cancelled"?"Stopped · best so far":"Highest among tested excerpts"}: ${packetLabels[job.mode]} · RMS error ${job.best_rmse.toExponential(3)} · ${job.successful} successful, ${job.failed} skipped, ${job.processed}/${job.total} tested. ${job.best_source.recording} · source offset ${time(job.best_source.offset)}. Only the winning sample was exported.`;
      $("status").textContent="Analysis ready";
    }else if(job.state==="cancelled"){
      $("error-search-result").textContent="Stopped before any sample could be analyzed.";$("status").textContent=analysis?"Analysis ready":"Ready";
    }else throw new Error(job.error||"No candidate could be analyzed");
  }catch(error){
    if(searchId)fetch(`/api/error-search/${searchId}/cancel`,{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"}).catch(()=>{});
    $("error").textContent=error.message;$("error").hidden=false;$("status").textContent="Search failed";
  }finally{searchId=null;setBusy(false);$("error-search-cancel").hidden=true;}
}
$("error-search-start").onclick=findHighestError;
$("error-search-cancel").onclick=async()=>{
  if(!searchId)return;$("error-search-cancel").disabled=true;
  try{const response=await fetch(`/api/error-search/${searchId}/cancel`,{method:"POST",headers:{"Content-Type":"application/json"},body:"{}"});if(!response.ok)throw new Error("Could not stop search");$("error-search-result").textContent="Stopping after the current excerpt; loading best so far…";}
  catch(error){$("error").textContent=error.message;$("error").hidden=false;$("error-search-cancel").disabled=false;}
};
window.addEventListener("pagehide",()=>{if(searchId)navigator.sendBeacon(`/api/error-search/${searchId}/cancel`,"{}");});
$("sample-form").addEventListener("submit",event=>{event.preventDefault();sample();});
$("shuffle").onclick=()=>{pinnedSample=null;const array=new Uint32Array(1);crypto.getRandomValues(array);$("seed").value=array[0];if($("sample-form").reportValidity())sample();};
$("dataset").onchange=()=>{pinnedSample=null;describe();};
$("sensitivity").oninput=()=>{$("sensitivity-value").value=`${Math.round(Number($("sensitivity").value)*100)}%`;};
$("outputs").onclick=event=>{const row=event.target.closest("tbody tr"),button=row?.querySelector("[data-track]");if(button)setTrack(button.dataset.track);};
$("spectral-view").onchange=drawSpectrum;$("pitch-candidates").onchange=drawAll;
$("full-features").addEventListener("toggle",drawAll);
audio.addEventListener("timeupdate",()=>{if(segmentEnd!==null&&audio.currentTime>=segmentEnd){audio.pause();segmentEnd=null;}drawWave();});
$("waveform").onclick=event=>{if(!analysis)return;segmentEnd=null;const rect=event.currentTarget.getBoundingClientRect();audio.currentTime=(event.clientX-rect.left)/rect.width*analysis.duration;drawWave();};
window.addEventListener("resize",drawAll);
async function init(){
  drawAll();
  try{const response=await fetch("/api/datasets");if(!response.ok)throw new Error("Could not load datasets");const data=await response.json();
    $("browse-dataset").hidden=!data.native_picker;
    updateDatasets(data.datasets,data.datasets.find(d=>d.import&&d.available)?.id);await sample();
  }catch(error){$("error").textContent=error.message;$("error").hidden=false;}
}
function registerTools(){
  if(!document.modelContext?.registerTool)return;
  const lifecycle=new AbortController();
  window.addEventListener("pagehide",()=>lifecycle.abort(),{once:true});
  const tools=[{
    name:"analyze_audio_sample",title:"Analyze an audio sample",
    description:"Sample a local dataset and update the voice segmentation player. Audio remains on this Mac.",
    inputSchema:{type:"object",properties:{dataset:{type:"string"},seed:{type:"integer"},stress:{type:"string",enum:["clean","noise20","noise10","noise0","reverb","hum"]}},required:["dataset","seed"],additionalProperties:false},
    annotations:{readOnlyHint:false,untrustedContentHint:true},
    async execute(input){
      if(!input||typeof input.dataset!=="string"||!Number.isInteger(input.seed)||!datasets.some(d=>d.id===input.dataset))throw new Error("Choose an available dataset and an integer seed");
      if(input.stress&&!['clean','noise20','noise10','noise0','reverb','hum'].includes(input.stress))throw new Error("Unknown listening condition");
      if(busy)throw new Error("An analysis is already running");
      $("dataset").value=input.dataset;$("seed").value=input.seed;if(input.stress)$("stress").value=input.stress;describe();
      return await sample();
    }
  },{
    name:"read_audio_analysis",title:"Read current audio analysis",
    description:"Read available datasets and the current segmentation result without changing playback.",
    inputSchema:{type:"object",properties:{},additionalProperties:false},
    annotations:{readOnlyHint:true,untrustedContentHint:true},
    execute(){return {datasets:datasets.map(d=>d.id),current:analysis?{artifact:analysis.artifact,source:analysis.source,segments:analysis.segments,stats:analysis.stats}:null};}
  }];
  for(const tool of tools){try{Promise.resolve(document.modelContext.registerTool(tool,{signal:lifecycle.signal})).catch(()=>{});}catch(error){console.info("Agent tools unavailable",error.message);}}
}
init().then(registerTools);
}
