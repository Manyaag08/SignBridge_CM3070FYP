/* ============================================================
   1 · signer profiles (localStorage)
   ============================================================ */
const STORE_KEY="signbridge.profiles.v2";
const DEFAULT_SLOTS=[
  {id:"how_are_you",   text:"How are you?",     how:"HOW: curved hands back-to-back, roll forward. YOU: point ahead."},
  {id:"i_am_good",     text:"I am good.",       how:"I: point to your chest. GOOD: flat hand from chin down into the other palm."},
  {id:"love_you",      text:"Love you.",        how:"LOVE: fists crossed on the chest. YOU: point ahead."},
  {id:"thank_you",     text:"Thank you.",       how:"Flat hand at the chin, move forward and down."},
  {id:"nice_to_meet",  text:"Nice to meet you.",how:"NICE: flat hand slides across the other palm. MEET: two index fingers come together."},
  {id:"my_name_is",    text:"My name is",       how:"MY: flat hand on the chest. NAME: two H-hands tap."},
  {id:"please",        text:"Please.",          how:"Flat hand circles on the chest."},
  {id:"sorry",         text:"Sorry.",           how:"A-hand circles on the chest."},
  {id:"help_me",       text:"Help me.",         how:"Thumbs-up on the flat palm, lift both towards you."},
  {id:"no_sign",       text:"No.",              how:"Index and middle finger tap the thumb."},
  {id:"see_you_later", text:"See you later.",   how:"SEE: V-hand from the eyes forward. LATER: L-hand rotates."},
  {id:"whats_up",      text:"What's up?",       how:"Both open hands, palms up, small shake."}
];
function blankProfile(){
  return {letters:{}, slots:JSON.parse(JSON.stringify(DEFAULT_SLOTS)), takes:{},
          opts:{hand:"right", personal:0.6, tau:0.22, builtin:true, tts:true, mirror:true}};
}
let store=null;
try{ store=JSON.parse(localStorage.getItem(STORE_KEY)||"null"); }catch(e){ store=null; }
if(!store||!store.profiles||!Object.keys(store.profiles).length){
  store={active:"Signer 1", profiles:{"Signer 1":blankProfile()}};
}
function P(){ return store.profiles[store.active]; }
let saveTimer=null;
function save(){
  clearTimeout(saveTimer);
  saveTimer=setTimeout(()=>{
    try{ localStorage.setItem(STORE_KEY,JSON.stringify(store)); }
    catch(e){ setStatus("Couldn't save profile — browser storage is full","err"); }
  },250);
}
// takes are stored as plain arrays; hydrate to Float32Array on use
const vecCache=new WeakMap();
function takeVecs(tk){
  let v=vecCache.get(tk);
  if(!v){ v=tk.f.map(r=>Float32Array.from(r)); vecCache.set(tk,v); }
  return v;
}
