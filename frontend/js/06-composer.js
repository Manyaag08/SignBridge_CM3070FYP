// 6 - message composer + speech out
let message="", lastEdit=0;   // lastEdit: when the message last changed (auto-send waits for a pause after it)
let letterProbs=[];          // one entry per character of `message`: the classifier's 26 probabilities for a typed letter, else null
function paintMessage(){ els("typed").textContent=message; lastEdit=performance.now(); }
function addChars(str,p){ message+=str; for(let i=0;i<str.length;i++) letterProbs.push(p||null); }
function appendLetter(ch,probs){ addChars(ch,probs); paintMessage(); }
function appendSpace(){ addChars(" "); paintMessage(); }
function backspace(){ message=message.slice(0,-1); letterProbs.pop(); paintMessage(); }
function clearMessage(){ message=""; letterProbs=[]; paintMessage(); }
function appendPhrase(txt){
  if(message && !/\s$/.test(message)) addChars(" ");
  addChars(txt+" ");
  paintMessage();
}
// turns the message into items the backend decoder wants: spelled words keep their per-letter probability
// vectors (that's what lets the dictionary decoder fix them), phrases/signed words just go through as text.
function messageItems(){
  const items=[]; let i=0;
  while(i<message.length){
    if(/\s/.test(message[i])){ i++; continue; }
    let j=i; while(j<message.length && !/\s/.test(message[j])) j++;
    const probs=letterProbs.slice(i,j);
    if(probs.every(Boolean)) items.push({probs:probs.map(p=>p.map(x=>+x.toFixed(4)))});
    else items.push({text:message.slice(i,j)});
    i=j;
  }
  return items;
}
// if we're in a call, send it to the other person instead of speaking it locally
// returns true when it went to a call partner
function speak(txt){
  if(callSend({type:"sign",text:txt})){ logMessage("me","sign",txt); return true; }
  speakLocal(txt);
  return false;
}
function speakLocal(txt){
  if(!window.speechSynthesis) return;
  try{
    const u=new SpeechSynthesisUtterance(txt.replace(/[^\w\s?.!,']/g,""));
    u.lang="en-US"; u.rate=1;
    speechSynthesis.cancel(); speechSynthesis.speak(u);
  }catch(e){}
}
let bannerTimer=null;
function showBanner(txt){
  els("bannerTxt").textContent=txt;
  els("banner").classList.add("on");
  clearTimeout(bannerTimer);
  bannerTimer=setTimeout(()=>els("banner").classList.remove("on"),2200);
}
let phraseLockUntil=0;
function flashRow(rowId){
  const el=rowId&&document.querySelector('[data-row="'+rowId+'"]');
  if(el){ el.classList.add("hit"); setTimeout(()=>el.classList.remove("hit"),1400); }
}
function firePhrase(txt,rowId){
  const now=performance.now();
  if(now<phraseLockUntil) return;
  phraseLockUntil=now+PH_LOCK_MS;
  lastLetter=null;
  appendPhrase(txt);
  showBanner(txt);
  if(P().opts.tts && !autoSendActive()) speak(txt);
  flashRow(rowId);
}

// ---- sign grammar: queue up glosses, hold off while a longer phrase might still come ----
const startsWith=(arr,pre)=>pre.length<=arr.length && pre.every((x,i)=>arr[i]===x);
const rowIdFor=text=>"bp_"+text.toLowerCase().replace(/[^a-z]+/g,"_");
function flushGlosses(){
  const out=[], q=glossQ;
  let i=0;
  while(i<q.length){
    const rest=q.slice(i);
    let best=null;
    for(const r of PHRASE_RULES) if(startsWith(rest,r.g) && (!best||r.g.length>best.g.length)) best=r;
    if(best){ out.push({text:best.text,g:best.g}); i+=best.g.length; }
    else { out.push({text:GLOSS_BY_ID[q[i]].word,g:[q[i]]}); i++; }
  }
  glossQ=[]; glossDeadline=0;
  return out;
}
function emitGlossPhrases(out,now){
  if(!out.length) return;
  out.forEach(o=>practicePhrase(o.text));
  if(practiceOnly()){ glossDone={g:out.flatMap(o=>o.g), text:out.map(o=>o.text).join(" "), until:now+2600}; return; }  // speaker practising: nothing typed or sent
  for(const o of out){ appendPhrase(o.text); flashRow(rowIdFor(o.text)); }
  const text=out.map(o=>o.text).join(" ");
  showBanner(text);
  if(P().opts.tts && !autoSendActive()) speak(text);
  glossDone={g:out.flatMap(o=>o.g), text, until:now+2600};
  phraseLockUntil=Math.max(phraseLockUntil,now+PH_LOCK_MS);
  lastLetter=null;
}
function onGloss(id,now){
  practiceGloss(id);
  phraseLockUntil=Math.max(phraseLockUntil,now+PH_LOCK_MS);
  lastLetter=null;
  const out=[];
  glossQ.push(id);
  if(glossQ.length>1 && !PHRASE_RULES.some(r=>startsWith(r.g,glossQ))){
    // this sign cannot continue the pending sequence: finish that one, start a new one
    glossQ.pop(); out.push(...flushGlosses()); glossQ=[id];
  }
  if(PHRASE_RULES.some(r=>r.g.length>glossQ.length && startsWith(r.g,glossQ))) glossDeadline=now+GLOSS_GAP_MS;
  else out.push(...flushGlosses());
  emitGlossPhrases(out,now);
  paintGloss(now);
}
let glossHTML="";
function paintGloss(now){
  let h;
  if(glossQ.length){
    h='signs: '+glossQ.map(g=>'<span class="g">'+g+'</span>').join(' → ')+
      ' <span class="wait">next sign… '+Math.max(0,(glossDeadline-now)/1000).toFixed(1)+'s</span>';
  }else if(glossDone && now<glossDone.until){
    h='signs: '+glossDone.g.map(g=>'<span class="g done">'+g+'</span>').join(' → ')+' → “'+glossDone.text+'”';
  }else h='signs: —';
  if(lastPointDir!==null)
    h+='<span class="dir">pointing '+(lastPointDir<-POINT_DIR?"at camera":lastPointDir>POINT_DIR?"at you":"sideways")+
       ' · '+lastPointDir.toFixed(2)+'</span>';
  if(h!==glossHTML){ glossHTML=h; els("gloss").innerHTML=h; }
}
setInterval(()=>{
  const now=performance.now();
  if(glossDeadline && now>=glossDeadline) emitGlossPhrases(flushGlosses(),now);
  paintGloss(now);
},150);
