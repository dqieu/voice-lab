"""Reference-bounded duration, boundary, pitch and formant evaluation."""
from __future__ import annotations

import numpy as np


def _mask(intervals, times, label=None):
    result=np.zeros(len(times),dtype=bool)
    for r in intervals:
        if label is None or r.get("label")==label:
            result |= (times>=r["start"])&(times<r["end"])
    return result


def _boundary_score(predicted, reference, tolerance):
    # Match only boundaries of the same type, once each, in chronological order.
    matched=0; errors=[]
    for kind in ("start","end"):
        p=sorted(r[kind] for r in predicted);q=sorted(r[kind] for r in reference)
        i=j=0
        while i<len(p) and j<len(q):
            difference=p[i]-q[j]
            if abs(difference)<=tolerance:
                matched+=1;errors.append(difference);i+=1;j+=1
            elif difference<0:
                i+=1
            else:
                j+=1
    precision=matched/(2*len(predicted)) if predicted else 0
    recall=matched/(2*len(reference)) if reference else 0
    return {"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0,
            "matched":matched,"median_absolute_error_ms":float(np.median(abs(np.array(errors)))*1000) if errors else None}


def evaluate(data, reference):
    """Intervals may include F0 base/modulation and known all-pole resonances.

    Reference tracks can instead be supplied as time/f0/formants arrays. No
    output from a second estimator is ever automatically treated as reference.
    """
    intervals=reference.get("intervals",[])
    times=np.arange(round(data["duration"]*1000))/1000
    true=_mask(intervals,times);pred=_mask(data["segments"],times)
    tp=int((true&pred).sum());fp=int((~true&pred).sum());fn=int((true&~pred).sum());tn=int((~true&~pred).sum())
    precision=tp/(tp+fp) if tp+fp else None;recall=tp/(tp+fn) if tp+fn else None
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    # Merge touching truth labels to score speech endpoints, not vowel boundaries.
    merged=[]
    for r in sorted(intervals,key=lambda r:r["start"]):
        if merged and r["start"]<=merged[-1]["end"]+.001:
            merged[-1]["end"]=max(merged[-1]["end"],r["end"])
        else:
            merged.append({"start":r["start"],"end":r["end"]})
    result={"reference_kind":reference.get("kind","user-supplied"),"resolution_ms":1,
            "activity":{"precision":precision,"recall":recall,"f1":f1,"confusion_seconds":{"tp":tp/1000,"fp":fp/1000,"fn":fn/1000,"tn":tn/1000},
                        "duration_error_fraction":(fp+fn)/max(1,len(times))},
            "boundaries":{str(ms):_boundary_score(data["segments"],merged,ms/1000) for ms in (10,20,30,50,100)}}
    t=np.array(data["frames"]["time"]);estimate=np.array(data["frames"]["f0"])
    truth=np.zeros(len(t));formant_truth=np.zeros((len(t),3))
    for r in intervals:
        selected=(t>=r["start"])&(t<r["end"])
        if "f0_hz" in r:
            truth[selected]=r["f0_hz"]+r.get("modulation_hz",0)*np.sin(2*np.pi*r.get("modulation_rate_hz",0)*(t[selected]-r["start"]+r.get("modulation_offset",0)))
        if "formants_hz" in r:
            formant_truth[selected]=r["formants_hz"][:3]
    tracks=reference.get("tracks")
    if tracks:
        ref_t=np.array(tracks["time"])
        if "f0" in tracks:
            # Nearest observation, no interpolation through voicing boundaries.
            i=np.searchsorted(ref_t,t).clip(0,len(ref_t)-1);previous=(i-1).clip(0,len(ref_t)-1)
            i=np.where(abs(ref_t[previous]-t)<abs(ref_t[i]-t),previous,i)
            truth=np.array(tracks["f0"])[i]
        if "formants" in tracks:
            formant_truth=np.stack([np.interp(t,ref_t,np.array(tracks["formants"])[:,j],left=0,right=0) for j in range(3)],axis=1)
    if np.any(truth>0):
        observed=(truth>0)&(estimate>0);voiced=truth>0
        relative=abs(estimate[observed]-truth[observed])/truth[observed]
        cents=1200*np.log2(estimate[observed]/truth[observed])
        result["pitch"]={"coverage":float(observed.sum()/voiced.sum()),"voicing_decision_error":float(np.mean((estimate>0)!=(truth>0))),
                         "mean_relative_error":float(relative.mean()) if len(relative) else None,
                         "median_absolute_cents":float(np.median(abs(cents))) if len(cents) else None,
                         "gross_error_threshold_relative":.2,"gross_error_rate":float(np.mean(relative>.2)) if len(relative) else None,
                         "octave_error_rate":float(np.mean((abs(cents)>1000)&(abs(cents)<1400))) if len(cents) else None}
    predicted=np.array(data["frames"]["formants"])
    if np.any(formant_truth>0):
        metrics=[]
        for j in range(3):
            available=formant_truth[:,j]>0;valid=available&(predicted[:,j]>0)
            error=predicted[valid,j]-formant_truth[valid,j]
            metrics.append({"mae_hz":float(np.mean(abs(error))) if len(error) else None,
                            "rmse_hz":float(np.sqrt(np.mean(error**2))) if len(error) else None,
                            "coverage":float(valid.sum()/available.sum()) if available.any() else None})
        result["formants"]=metrics
    return result
