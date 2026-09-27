/* ============================================================
   8 · profile UI wiring
   ============================================================ */
function refreshProfileUI(){
  const sel=els("profSel"); sel.innerHTML="";
  for(const name of Object.keys(store.profiles)){
    const o=document.createElement("option"); o.value=name; o.textContent=name;
    if(name===store.active) o.selected=true;
    sel.appendChild(o);
  }
  const prof=P(), o=prof.opts;
  els("handSel").value=o.hand;
  els("persRange").value=Math.round(o.personal*100);
  els("persVal").textContent=Math.round(o.personal*100)+"%";
  els("tauRange").value=Math.round(o.tau*100);
  els("tauVal").textContent=o.tau.toFixed(2);
  els("builtinChk").checked=!!o.builtin;
  els("ttsChk").checked=!!o.tts;
  els("mirrorChk").checked=!!o.mirror;
  const nT=Object.keys(prof.takes).reduce((a,k)=>a+prof.takes[k].length,0);
  els("profInfo").textContent=Object.keys(store.profiles).length+" profile"+(Object.keys(store.profiles).length>1?"s":"")+" · "+nT+" phrase recording"+(nT===1?"":"s");
  refreshRefBadges(); buildPhraseLists(); refreshCal();
}
els("profSel").onchange=e=>{ store.active=e.target.value; save(); resetRuntime(); refreshProfileUI(); };
els("profNew").onclick=()=>{
  const name=prompt("Name for the new signer profile:","Signer "+(Object.keys(store.profiles).length+1));
  if(!name) return;
  if(store.profiles[name]){ alert("That profile already exists."); return; }
  store.profiles[name]=blankProfile(); store.active=name; save(); resetRuntime(); refreshProfileUI();
};
els("profDel").onclick=()=>{
  const names=Object.keys(store.profiles);
  if(names.length<2){ alert("Keep at least one profile."); return; }
  if(!confirm('Delete profile "'+store.active+'" and all of its calibration data?')) return;
  delete store.profiles[store.active];
  store.active=Object.keys(store.profiles)[0];
  save(); resetRuntime(); refreshProfileUI();
};
els("handSel").onchange=e=>{ P().opts.hand=e.target.value; save(); };
els("persRange").oninput=e=>{
  P().opts.personal=+e.target.value/100;
  els("persVal").textContent=e.target.value+"%"; save();
};
els("tauRange").oninput=e=>{
  P().opts.tau=+e.target.value/100;
  els("tauVal").textContent=P().opts.tau.toFixed(2); save();
};
els("builtinChk").onchange=e=>{ P().opts.builtin=e.target.checked; save(); };
els("ttsChk").onchange=e=>{ P().opts.tts=e.target.checked; save(); };
els("mirrorChk").onchange=e=>{ P().opts.mirror=e.target.checked; save(); };

els("expBtn").onclick=()=>{
  const blob=new Blob([JSON.stringify({active:store.active,profiles:store.profiles},null,1)],
    {type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob);
  a.download="signbridge_profiles.json";
  a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),2000);
};
els("impBtn").onclick=()=>els("impFile").click();
els("impFile").onchange=e=>{
  const f=e.target.files[0]; if(!f) return;
  const r=new FileReader();
  r.onload=()=>{
    try{
      const d=JSON.parse(r.result);
      if(!d.profiles) throw new Error("no profiles");
      for(const k in d.profiles){
        const p=d.profiles[k];
        if(!p.opts) p.opts=blankProfile().opts;
        if(!p.slots) p.slots=JSON.parse(JSON.stringify(DEFAULT_SLOTS));
        if(!p.letters) p.letters={};
        if(!p.takes) p.takes={};
        store.profiles[store.profiles[k]?k+" (imported)":k]=p;
      }
      save(); refreshProfileUI();
      alert("Imported "+Object.keys(d.profiles).length+" profile(s).");
    }catch(err){ alert("That file is not a SignBridge profile export."); }
  };
  r.readAsText(f);
  e.target.value="";
};
