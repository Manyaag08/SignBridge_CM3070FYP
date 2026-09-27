/* ============================================================
   17 · learn & practise: pick a letter, built-in sign or phrase, watch the avatar demonstrate it, then sign it.
   "Matched" is only ever shown once the live recogniser has actually read the target:
     letters  - renderReadout() (11-main-loop.js) calls practiceTick() every frame; the target letter must be the
                top guess at >= CONF for PRAC_HOLD_MS without a break
     signs    - onGloss() (06-composer.js) calls practiceGloss() when a built-in sign is confirmed
     phrases  - emitGlossPhrases() calls practicePhrase() with the text the sign grammar produced
   On the speaker's screen detection runs for practice only (practiceOnly()): nothing is typed, spoken or sent.
   ============================================================ */
const PRAC_HOLD_MS=700;
const PRAC_ITEMS={
  letters: LABELS.map(L=>({kind:"letter",key:L,label:L,how:"Fingerspell the letter "+L+"."})),
  signs:   GLOSSES.map(G=>({kind:"sign",key:G.id,label:G.id,how:G.how,G})),
  phrases: (()=>{ const m=new Map();
    for(const r of PHRASE_RULES){ if(!m.has(r.text)) m.set(r.text,[]); m.get(r.text).push(r.g); }
    return [...m].map(([text,seqs])=>({kind:"phrase",key:text,label:text,seqs,
      how:"Sign "+seqs[0].join(" → ")+(seqs.length>1?" (or "+seqs.slice(1).map(s=>s.join(" → ")).join(", ")+")":"")+", one after another."}));
  })()
};
const prac={mode:"letters", idx:0, since:0, done:false, matched:0, seen:[]};
function practiceOnly(){ return document.body.classList.contains("role-speech"); }
function pracItem(){ return PRAC_ITEMS[prac.mode][prac.idx]; }

function practiceShow(i){
  const list=PRAC_ITEMS[prac.mode];
  prac.idx=(i+list.length)%list.length; prac.since=0; prac.done=false; prac.seen=[];
  const it=pracItem();
  els("pracLetter").textContent=it.label;
  els("pracLetter").classList.toggle("long",it.label.length>3);
  els("pracHow").textContent=it.how;
  els("pracGlyph").innerHTML=it.kind==="letter"?skeletonSVG(REF[it.key]):it.kind==="sign"?glossGlyph(it.G):phraseGlyphs(it.seqs);
  pracSetState("idle","Not matched yet");
}
function pracSetState(state,msg){
  const t=els("pracTarget");
  t.classList.toggle("ok",state==="ok"); t.classList.toggle("near",state==="near");
  els("pracFeedback").textContent=msg;
  if(state!=="near") els("pracHold").style.width=state==="ok"?"100%":"0%";
}
function pracMatched(){
  if(prac.done) return;
  const it=pracItem();
  prac.done=true; prac.matched++;
  els("pracStreak").textContent=prac.matched+" matched";
  pracSetState("ok","✓ Matched — "+(it.kind==="letter"?"the letter "+it.label:"“"+(it.G?it.G.word:it.label)+"”"));
  const mode=prac.mode, idx=prac.idx;
  setTimeout(()=>{ if(prac.done && prac.mode===mode && prac.idx===idx) practiceShow(idx+1); },1800);
}
function practiceTick(best){
  if(prac.done || prac.mode!=="letters") return;
  const L=pracItem().key, now=performance.now();
  if(!best){ prac.since=0; pracSetState("idle","Not matched yet"); return; }
  if(best.letter===L && best.p>=CONF){
    if(!prac.since) prac.since=now;
    const f=Math.min(1,(now-prac.since)/PRAC_HOLD_MS);
    els("pracHold").style.width=Math.round(f*100)+"%";
    if(f>=1) pracMatched(); else pracSetState("near","Almost — hold it steady…");
  }else{
    prac.since=0;
    pracSetState("idle",best.p>=CONF?"Not matched — that reads as "+best.letter:"Not matched yet");
  }
}
function practiceGloss(id){
  if(prac.done) return;
  const it=pracItem();
  if(it.kind==="sign"){
    if(id===it.key) pracMatched(); else pracSetState("idle","Not matched — that was "+id);
  }else if(it.kind==="phrase"){
    prac.seen.push(id);
    pracSetState("near","Seen "+prac.seen.slice(-4).join(" → ")+" …");
  }
}
function practicePhrase(text){
  if(prac.done) return;
  const it=pracItem();
  if(it.kind!=="phrase") return;
  if(text===it.key) pracMatched();
  else { prac.seen=[]; pracSetState("idle","Not matched — that read as “"+text+"”"); }
}
function practiceDemo(){
  const it=pracItem();
  if(it.kind==="letter") avatarSay(it.label,[{type:"spell",value:it.key}]);
  else avatarSay(it.kind==="sign"?it.G.word:it.key);
}
document.querySelectorAll("#pracModes button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("#pracModes button").forEach(x=>x.classList.toggle("on",x===b));
  prac.mode=b.dataset.mode; practiceShow(0);
});
els("pracPrev").onclick=()=>practiceShow(prac.idx-1);
els("pracNext").onclick=()=>practiceShow(prac.idx+1);
els("pracShow").onclick=practiceDemo;
els("pracCamBtn").onclick=()=>camOn?stopCamera():start();

// the avatar is a single canvas that lives in the call (signer) or the preview card (speaker); mirror it here so
// the demonstration is visible right next to the practice target on either screen
(function mirrorAvatar(){
  const src=els("avatar"), dst=els("pracAvatar"), c=dst.getContext("2d");
  (function draw(){ c.clearRect(0,0,dst.width,dst.height); c.drawImage(src,0,0,dst.width,dst.height); requestAnimationFrame(draw); })();
})();
practiceShow(0);
