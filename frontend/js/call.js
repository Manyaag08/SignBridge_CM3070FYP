/* ============================================================
   Call layer: a two-person WebRTC video call. The media travels peer to peer; the backend only relays the small
   signalling messages (offer / answer / ICE) over a WebSocket and never sees audio or video. A WebRTC data channel
   next to the media carries SignBridge's own messages:
     {type:"sign",   text}            the Deaf caller's recognised signs, as a sentence → spoken aloud on the other side
     {type:"speech", text, tone}      the hearing caller's Whisper transcript + tone   → captions + signing avatar
   Both callers can send both kinds, so either side can be the signer.
   ============================================================ */
const call={ws:null,pc:null,dc:null,local:null,room:"",pending:[],pingTimer:null,rtt:null};

function callSend(m){
  if(call.dc&&call.dc.readyState==="open"){ call.dc.send(JSON.stringify(m)); return true; }
  return false;
}
function callOpen(){ return !!(call.dc&&call.dc.readyState==="open"); }
function callActive(){ return !!call.ws; }

// one camera for everything: the call sends this stream and sign detection (11-main-loop.js) reads it, so the
// self-view, what the partner sees and what the recogniser sees are the same pictures.
async function getLocalStream(){
  if(call.local && call.local.getVideoTracks().some(t=>t.readyState==="live")) return call.local;
  try{ call.local=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480},audio:true}); }
  catch(e){ call.local=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480}}); }   // no microphone
  els("localVideo").srcObject=call.local; els("pipOff").style.display="none";
  return call.local;
}
function releaseLocalStream(){
  if(call.local){ call.local.getTracks().forEach(t=>t.stop()); call.local=null; }
  els("localVideo").srcObject=null; els("pipOff").style.display="";
}

// conversation log, both sides, oldest first like a chat
function logMessage(who,kind,text,tone){
  const box=els("remoteMsgs");
  const row=document.createElement("div");
  row.className="rm "+kind+" "+who;
  const label=document.createElement("span"); label.className="rmwho";
  label.textContent=(who==="me"?"You ":"Partner ")+(kind==="sign"?"signed":"said");
  const body=document.createElement("span"); body.className="rmtext"; body.textContent="“"+text+"”";
  row.append(label,body);
  if(tone){ const t=document.createElement("span"); t.className="tone "+tone; t.textContent=tone; row.append(" ",t); }
  box.appendChild(row);
  while(box.children.length>40) box.removeChild(box.firstChild);
  box.scrollTop=box.scrollHeight;
  return row;
}
function callStatus(t,state){
  // same three states as setStatus() in 11-main-loop.js: true/"ok" connected, "err" a real failure, left out
  // for "still working on it" (opening camera, waiting for the other person, ...)
  els("callStatus").innerHTML='<span class="dot"></span>'+t;
  els("callStatus").className="status"+(state===true||state==="ok"?" ok":state==="err"?" err":"");
}
function sigSend(m){ if(call.ws&&call.ws.readyState===1) call.ws.send(JSON.stringify(m)); }

function makePc(){
  closePc();
  // no STUN/TURN by default: candidates stay on this machine / LAN, consistent with the privacy goal.
  // For two computers on different networks add {urls:"stun:..."} here.
  call.pc=new RTCPeerConnection({iceServers:[]});
  call.local.getTracks().forEach(t=>call.pc.addTrack(t,call.local));
  call.pc.onicecandidate=e=>{ if(e.candidate) sigSend({type:"candidate",candidate:e.candidate}); };
  call.pc.ontrack=e=>{ els("remoteVideo").srcObject=e.streams[0]; };
  call.pc.ondatachannel=e=>wireDc(e.channel);
  call.pc.onconnectionstatechange=()=>{
    const s=call.pc&&call.pc.connectionState;
    if(s==="connected") callStatus("Connected · room “"+call.room+"”",true);
    else if(s==="failed"||s==="disconnected") callStatus("Connection "+s,"err");
  };
  call.pending=[];
}
function closePc(){
  clearInterval(call.pingTimer);
  if(call.dc){ try{call.dc.close();}catch(e){} call.dc=null; }
  if(call.pc){ try{call.pc.close();}catch(e){} call.pc=null; }
  els("remoteVideo").srcObject=null;
}
function wireDc(dc){
  call.dc=dc;
  dc.onopen=()=>{
    callStatus("Connected · room “"+call.room+"”",true);
    // a signer's recognition then runs on the call camera with no separate step. It starts once the call is up,
    // not while it is connecting: loading the hand tracker and classifier briefly blocks the page, and doing
    // that during the offer/answer exchange delayed the connection itself by several seconds.
    if(document.body.classList.contains("role-sign") && typeof start==="function") start();
    call.pingTimer=setInterval(()=>callSend({type:"ping",t:performance.now()}),2000);
  };
  dc.onclose=()=>clearInterval(call.pingTimer);
  dc.onmessage=e=>{ let m; try{ m=JSON.parse(e.data); }catch(err){ return; } onRemote(m); };
}
// the tone tag for the most recently received "speech" message arrives as its own late follow-up (see
// 12-speech.js onUtterance) so the message itself - and the avatar signing it - doesn't wait on it.
let lastSpeechRow=null, lastSpeechText=null;
function onRemote(m){
  if(m.type==="ping"){ callSend({type:"pong",t:m.t}); return; }
  if(m.type==="pong"){ call.rtt=performance.now()-m.t; els("callRtt").textContent="Latency "+call.rtt.toFixed(0)+" ms"; return; }
  if(m.type==="tone"){
    if(m.tone && lastSpeechRow && lastSpeechText===m.text && !lastSpeechRow.querySelector(".tone")){
      const t=document.createElement("span"); t.className="tone "+m.tone; t.textContent=m.tone;
      lastSpeechRow.append(" ",t);
    }
    return;
  }
  if(m.type==="sign"){
    logMessage("partner","sign",m.text);
    speakLocal(m.text);                                       // the hearing side hears the Deaf caller, automatically
  }else if(m.type==="speech"){
    lastSpeechRow=logMessage("partner","speech",m.text,m.tone); lastSpeechText=m.text;
    if(els("avAuto").checked) avatarSay(m.text);              // the Deaf side sees captions + the avatar signing it
  }
}

async function joinCall(){
  const room=els("roomInput").value.trim()||"demo";
  els("joinBtn").disabled=true; callStatus("Starting camera and microphone…");
  try{ await getLocalStream(); }
  catch(e){ callStatus("Camera or microphone blocked","err"); els("joinBtn").disabled=false; return; }
  call.room=room;
  call.ws=new WebSocket((location.protocol==="https:"?"wss://":"ws://")+location.host+"/ws/"+encodeURIComponent(room));
  call.ws.onopen=()=>callStatus("Waiting for your partner in “"+room+"”…");
  call.ws.onerror=()=>{ callStatus("Couldn't reach the call server — is SignBridge running?","err"); els("joinBtn").disabled=false; };
  call.ws.onmessage=async ev=>{
    const m=JSON.parse(ev.data);
    if(m.type==="peer-joined"){                                   // someone arrived after me: I make the offer
      makePc(); wireDc(call.pc.createDataChannel("signbridge"));
      const offer=await call.pc.createOffer(); await call.pc.setLocalDescription(offer);
      sigSend({type:"offer",sdp:call.pc.localDescription});
    }else if(m.type==="offer"){
      makePc();
      await call.pc.setRemoteDescription(m.sdp);
      for(const c of call.pending) await call.pc.addIceCandidate(c); call.pending=[];
      const ans=await call.pc.createAnswer(); await call.pc.setLocalDescription(ans);
      sigSend({type:"answer",sdp:call.pc.localDescription});
    }else if(m.type==="answer"){
      await call.pc.setRemoteDescription(m.sdp);
      for(const c of call.pending) await call.pc.addIceCandidate(c); call.pending=[];
    }else if(m.type==="candidate"){
      if(call.pc&&call.pc.remoteDescription) await call.pc.addIceCandidate(m.candidate); else call.pending.push(m.candidate);
    }else if(m.type==="peer-left"){
      closePc(); callStatus("Your partner left — waiting for them to rejoin…");
    }else if(m.type==="room-full"){
      callStatus("That room is full — try another room name","err"); leaveCall();
    }
  };
  els("joinBtn").style.display="none"; els("leaveBtn").style.display="";
}
function leaveCall(){
  closePc();
  if(call.ws){ try{call.ws.close();}catch(e){} call.ws=null; }
  if(!(typeof camOn!=="undefined" && camOn)) releaseLocalStream();   // keep it if sign detection is still using it
  els("joinBtn").style.display=""; els("joinBtn").disabled=false; els("leaveBtn").style.display="none";
  els("callRtt").textContent="Latency —";
  callStatus("Not connected");
}
els("joinBtn").onclick=joinCall;
els("leaveBtn").onclick=leaveCall;
