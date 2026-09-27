// 16 - role selection / screen routing
// app opens on a landing screen where you pick how you're using it. that choice just locks the call
// screen to one pipeline: "I sign" -> camera/classifier/composer + the incoming-speech avatar,
// "I speak" -> microphone/transcript. both sides still share the same call layer (call.js) and
// conversation log, so either side still gets whatever the other side sends - the role only decides
// which tools *this* browser shows for typing stuff in. doesn't touch call.js, the classifier or the
// speech pipeline at all, just toggles which panels are visible and which of the two screens is on.
const ROLE_KEY="signbridge.role";
let currentRole=null;

function showScreen(id){
  document.querySelectorAll(".screen").forEach(s=>s.classList.toggle("on",s.id===id));
}

function selectRole(role){
  currentRole=role;
  try{ localStorage.setItem(ROLE_KEY,role); }catch(e){}
  document.body.classList.remove("role-sign","role-speech");
  document.body.classList.add(role==="sign"?"role-sign":"role-speech");
  els("roleBadge").textContent=role==="sign"?"signing":"speaking";
  // the avatar panel is shown on both screens now, but it signs different things depending on which
  // side you're on - the partner's speech on the signer's screen, your own on the speaker's - so the
  // heading has to say which one you're actually looking at.
  els("avatarTitle").textContent=role==="sign"?"Partner's speech, signed for you":"Your speech, signed";
  // only the signing partner needs the avatar during the call; on the speaker's screen it moves below the call
  // as a testing preview of how their speech is signed
  const card=document.querySelector(".avatarcard");
  if(role==="sign") els("sidePanel").prepend(card); else els("avatarPreviewSlot").appendChild(card);
  document.body.classList.add("in-call");
  showScreen("screen-call");
  window.scrollTo(0,0);
}

function backToLanding(){
  if(typeof callActive==="function" && callActive()) leaveCall();
  if(typeof stopCamera==="function") stopCamera();
  document.body.classList.remove("in-call");
  showScreen("screen-landing");
  window.scrollTo(0,0);
}

els("roleSignBtn").onclick=()=>selectRole("sign");
els("roleSpeechBtn").onclick=()=>selectRole("speech");
els("changeRoleBtn").onclick=backToLanding;

// always start back on the landing screen - a returning signer has to pick their role again instead of
// getting dropped straight into a call, but I pre-highlight whatever they picked last time as a shortcut.
(function preselectLastRole(){
  let last=null;
  try{ last=localStorage.getItem(ROLE_KEY); }catch(e){}
  if(last==="sign") els("roleSignBtn").textContent="Continue as a signer · last used";
  else if(last==="speech") els("roleSpeechBtn").textContent="Continue as a speaker · last used";
})();
showScreen("screen-landing");
