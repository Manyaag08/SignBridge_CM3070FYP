/* ============================================================
   12 · Pipeline B — speech → text → tone
   Primary path: microphone audio is segmented by a small energy-based voice detector in the browser, each utterance is
   encoded as 16 kHz mono WAV and sent to the local backend, where Whisper transcribes it (audio never leaves the
   machine) and the tone tagger labels it. Fallback when the backend is not running: the browser's Web Speech API, which
   in Chrome uses a cloud service, so it is labelled as such in the interface.
   ============================================================ */
const utts=[];                       // [{text, tone, score, ms}]
let interimText="", recOn=false, engine=null;

function srStatus(t,cls){
  els("srstatus").innerHTML='<span class="dot"></span>'+t;
  els("srstatus").className="status"+(cls?" ok":"");
  els("srpill").textContent=cls==="rec"?"● Live":(recOn?"Live":"Off");
  els("srpill").className="pill"+(cls==="rec"?" rec":"");
}
const esc=s=>String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function paintTranscript(){
  if(!utts.length && !interimText){
    els("transcript").innerHTML='<span class="interim">Press “Start speaking” and talk…</span>'; return;
  }
  els("transcript").innerHTML=utts.map(u=>esc(u.text)+
      (u.tone?' <span class="tone '+u.tone+'" title="tone tagger score '+(u.score!=null?u.score.toFixed(2):"")+'">'+u.tone+'</span>':"")).join("\n")+
    (interimText?'\n<span class="interim">'+esc(interimText)+'</span>':"");
  els("transcript").scrollTop=els("transcript").scrollHeight;
}

// one finished utterance from either engine: show it, sign it locally, send it across the call straight
// away, then tag its tone. The tone tag is sent as a separate follow-up patch (see call.js) rather than
// being awaited first - waiting on that extra network round trip before sending was adding a whole
// tone-tagging request's worth of delay before the *other* caller's avatar even started signing.
async function onUtterance(text,ms){
  text=(text||"").trim(); if(!text) return;
  const u={text,tone:null,score:null,ms};
  utts.push(u); paintTranscript();
  if(els("avAuto").checked) avatarSay(text);
  if(callSend({type:"speech",text:u.text,tone:null,score:null})) u.row=logMessage("me","speech",u.text);
  if(API.ok){
    try{
      const t=await API.tone(text); u.tone=t.tone; u.score=t.score; paintTranscript();
      callSend({type:"tone",text:u.text,tone:u.tone,score:u.score});
      if(u.row && u.tone && !u.row.querySelector(".tone")){ const t=document.createElement("span"); t.className="tone "+u.tone; t.textContent=u.tone; u.row.append(" ",t); }
    }catch(e){ console.warn("tone failed",e); }
  }
}

/* ---- engine 1: Whisper through the backend ---- */
function encodeWav(f32,inRate,outRate){
  let data=f32;
  if(inRate!==outRate){                                     // linear-interpolation resample to 16 kHz
    const n=Math.floor(f32.length*outRate/inRate); data=new Float32Array(n);
    for(let i=0;i<n;i++){ const x=i*inRate/outRate, j=Math.floor(x), f=x-j; data[i]=(f32[j]||0)*(1-f)+(f32[j+1]||0)*f; }
  }
  const buf=new ArrayBuffer(44+data.length*2), v=new DataView(buf);
  const w=(o,s)=>{ for(let i=0;i<s.length;i++) v.setUint8(o+i,s.charCodeAt(i)); };
  w(0,"RIFF"); v.setUint32(4,36+data.length*2,true); w(8,"WAVE"); w(12,"fmt "); v.setUint32(16,16,true);
  v.setUint16(20,1,true); v.setUint16(22,1,true); v.setUint32(24,outRate,true); v.setUint32(28,outRate*2,true);
  v.setUint16(32,2,true); v.setUint16(34,16,true); w(36,"data"); v.setUint32(40,data.length*2,true);
  for(let i=0;i<data.length;i++){ const s=Math.max(-1,Math.min(1,data[i])); v.setInt16(44+i*2,s<0?s*0x8000:s*0x7fff,true); }
  return new Blob([buf],{type:"audio/wav"});
}
const BackendSTT={
  ctx:null, stream:null, node:null, src:null, queue:Promise.resolve(),
  pre:[], seg:[], inSpeech:false, silentMs:0, speechMs:0, noise:0.004,
  // HANG_MS was 700ms: long enough that a whole sentence would queue up before Whisper (and so the avatar)
  // ever saw a word of it, making the signing look like it started well "after" the sentence was spoken.
  // 450ms still clears a normal mid-sentence breath but chops speech into shorter, more frequent utterances,
  // so each one reaches transcription - and the avatar - sooner.
  MAX_S:20, HANG_MS:450, MIN_SPEECH_MS:350, PRE_MS:300,
  async start(){
    this.stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,channelCount:1}});
    this.ctx=new (window.AudioContext||window.webkitAudioContext)();
    this.src=this.ctx.createMediaStreamSource(this.stream);
    this.node=this.ctx.createScriptProcessor(4096,1,1);
    this.node.onaudioprocess=e=>this.frame(new Float32Array(e.inputBuffer.getChannelData(0)));
    this.src.connect(this.node); this.node.connect(this.ctx.destination);   // output buffer is left silent
    this.reset();
  },
  reset(){ this.pre=[]; this.seg=[]; this.inSpeech=false; this.silentMs=0; this.speechMs=0; },
  frame(x){
    const ms=x.length/this.ctx.sampleRate*1000;
    let e=0; for(let i=0;i<x.length;i++) e+=x[i]*x[i];
    const rms=Math.sqrt(e/x.length);
    const speaking=rms>Math.max(0.012,this.noise*3);
    if(!speaking && !this.inSpeech) this.noise=this.noise*0.95+rms*0.05;     // adapt to room noise while idle
    if(speaking){
      if(!this.inSpeech){ this.inSpeech=true; this.seg=this.pre.slice(); this.speechMs=0; }
      this.silentMs=0; this.speechMs+=ms; this.seg.push(x);
    }else if(this.inSpeech){
      this.seg.push(x); this.silentMs+=ms;
      if(this.silentMs>=this.HANG_MS) this.flush();
    }else{
      this.pre.push(x);
      while(this.pre.length*ms>this.PRE_MS) this.pre.shift();
    }
    if(this.inSpeech && this.seg.length*ms>this.MAX_S*1000) this.flush();
    els("micLevel").style.width=Math.min(100,Math.round(rms*600))+"%";
  },
  flush(){
    const seg=this.seg, ok=this.speechMs>=this.MIN_SPEECH_MS; this.reset();
    if(!ok) return;
    let n=0; for(const c of seg) n+=c.length;
    const all=new Float32Array(n); let o=0; for(const c of seg){ all.set(c,o); o+=c.length; }
    const wav=encodeWav(all,this.ctx.sampleRate,16000);
    srStatus("Transcribing…","rec");
    this.queue=this.queue.then(async()=>{
      const t0=performance.now();
      try{
        const r=await API.transcribe(wav,els("langSel").value);
        await onUtterance(r.text,performance.now()-t0);
      }catch(e){ srStatus("Transcription error: "+e.message); console.error(e); }
      if(recOn) srStatus("Listening — transcribed privately on this device","rec");
    });
  },
  stop(){
    try{ this.node&&this.node.disconnect(); this.src&&this.src.disconnect(); }catch(e){}
    if(this.stream) this.stream.getTracks().forEach(t=>t.stop());
    if(this.ctx) this.ctx.close();
    this.ctx=this.stream=this.node=this.src=null; els("micLevel").style.width="0%";
  }
};

/* ---- engine 2: Web Speech API fallback (cloud service in Chrome) ---- */
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
let webRec=null;
// the browser's own recogniser is the one engine here with real streaming partial results, so it's the
// only one that can sign genuinely as-you-speak rather than waiting for a finished utterance: each time the
// interim transcript grows, only the newly-added words are sent to the avatar, so it keeps pace with speech
// instead of replaying the whole sentence from the start on every update.
let signedInterimWords=0;
const WebSpeechSTT={
  start(){
    webRec=new SR(); webRec.continuous=true; webRec.interimResults=true; webRec.lang=els("langSel").value==="auto"?"en-US":els("langSel").value;
    signedInterimWords=0;
    webRec.onresult=e=>{
      let interim="";
      for(let i=e.resultIndex;i<e.results.length;i++){
        const tr=e.results[i][0].transcript;
        if(e.results[i].isFinal){ interimText=""; signedInterimWords=0; onUtterance(tr,0); } else interim+=tr;
      }
      interimText=interim; paintTranscript();
      if(interim && els("avAuto").checked){
        const words=interim.trim().split(/\s+/).filter(Boolean);
        if(words.length>signedInterimWords){
          avatarSay(words.slice(signedInterimWords).join(" "));
          signedInterimWords=words.length;
        }
      }
    };
    webRec.onerror=e=>srStatus("Speech error: "+e.error);
    webRec.onend=()=>{ if(recOn&&engine===WebSpeechSTT) webRec.start(); };
    webRec.start();
  },
  stop(){ try{ webRec.stop(); }catch(e){} }
};

els("micBtn").onclick=async()=>{
  if(!recOn){
    await API.health();
    engine=API.ok?BackendSTT:(SR?WebSpeechSTT:null);
    if(!engine){ srStatus("No speech engine available — start SignBridge or use Chrome"); return; }
    try{ await engine.start(); }catch(e){ srStatus("Microphone access blocked"); console.error(e); return; }
    recOn=true;
    els("micBtn").textContent="Stop speaking"; els("micBtn").classList.remove("blue");
    srStatus(engine===BackendSTT?"Listening — transcribed privately on this device":"Listening — using the browser's speech service (not private)","rec");
  }else{
    recOn=false; engine.stop(); srStatus("Microphone off");
    els("micBtn").textContent="Start speaking"; els("micBtn").classList.add("blue");
  }
};
els("clearSpeech").onclick=()=>{ utts.length=0; interimText=""; paintTranscript(); };
paintTranscript();
