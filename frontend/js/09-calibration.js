/* ============================================================
   9 · calibration capture
   ============================================================ */
let calState=null;   // {letter, need, got:[]}
function refreshCal(){
  const L=els("calLetter").value, n=(P().letters[L]||[]).length;
  els("calStatus").innerHTML='<span class="dot"></span>'+
    (n? "Letter "+L+": "+n+" samples saved" : "Letter "+L+": no samples yet");
  els("calStatus").className="status"+(n?" ok":"");
}
els("calLetter").onchange=refreshCal;
els("calBtn").onclick=()=>{
  if(!session){ alert("Start the camera first."); return; }
  const L=els("calLetter").value;
  // 12 frames is enough to average out a shaky hold without asking someone to pose for ages
  countdown(3,()=>{ calState={letter:L,need:12,got:[]}; els("calBtn").disabled=true; });
};
els("calClear").onclick=()=>{ delete P().letters[els("calLetter").value]; save(); refreshRefBadges(); refreshCal(); };
els("calClearAll").onclick=()=>{
  if(!confirm("Clear every calibrated letter for this profile?")) return;
  P().letters={}; save(); refreshRefBadges(); refreshCal();
};
function calTick(feat){
  if(!calState) return;
  calState.got.push(Array.from(feat).map(v=>+v.toFixed(4)));
  els("calStatus").innerHTML='<span class="dot"></span>Capturing '+calState.letter+
    ' · '+calState.got.length+'/'+calState.need;
  if(calState.got.length>=calState.need){
    const lib=P().letters, L=calState.letter;
    // cap at 24 so recalibrating a letter several times doesn't quietly grow this forever - keeps only the
    // most recent takes, which also lets someone "fix" a bad calibration by just recording it again
    lib[L]=(lib[L]||[]).concat(calState.got).slice(-24);
    calState=null; els("calBtn").disabled=false;
    save(); refreshRefBadges(); refreshCal();
  }
}
