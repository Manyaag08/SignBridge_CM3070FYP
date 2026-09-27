/* ============================================================
   4 · built-in geometric signs
   ============================================================ */
function geometry(feat){
  const pt=i=>[feat[i*2],feat[i*2+1]];
  const d=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
  const W=[0,0];
  const ext=name=>{ const f=FING[name]; return d(pt(f[3]),W) > d(pt(f[1]),W)*1.12; };
  // how straight a finger is: knuckle→tip distance over the length of its three bones.
  // ~1.0 when straight, ~0.85 for a curved C-hand, ~0.75 for a closed O-hand.
  const straight=name=>{
    const f=FING[name];
    const chain=d(pt(f[0]),pt(f[1]))+d(pt(f[1]),pt(f[2]))+d(pt(f[2]),pt(f[3]));
    return chain? d(pt(f[0]),pt(f[3]))/chain : 0;
  };
  const thumb = d(pt(4),pt(17)) > d(pt(2),pt(17))*1.15;
  const i=ext("index"), m=ext("middle"), r=ext("ring"), p=ext("pinky");
  return {
    thumb, index:i, middle:m, ring:r, pinky:p,
    ily: thumb && i && p && !m && !r && straight("index")>0.9 && straight("pinky")>0.85,
    // the thumb must stand clear above a folded index, which rules out O and F
    thumbsUp: thumb && !i && !m && !r && !p &&
              pt(4)[1] < -0.55 && pt(4)[1] < pt(8)[1]-0.15 && d(pt(4),pt(8)) > 0.45,
    // straightness separates a flat palm from the curved C-hand
    openPalm: thumb && i && m && r && p &&
              straight("index")>0.93 && straight("middle")>0.93 && straight("ring")>0.93,
    fist: !thumb && !i && !m && !r && !p
  };
}
function oscillation(hist,axis){
  if(hist.length<6) return {amp:0,changes:0};
  const v=hist.map(h=>axis==="x"?h.x:h.y);
  let mn=Infinity,mx=-Infinity;
  for(const q of v){ if(q<mn)mn=q; if(q>mx)mx=q; }
  let changes=0,dir=0;
  for(let k=1;k<v.length;k++){
    const dv=v[k]-v[k-1];
    if(Math.abs(dv)<0.03) continue;
    const nd=dv>0?1:-1;
    if(dir&&nd!==dir) changes++;
    dir=nd;
  }
  return {amp:mx-mn,changes};
}
// 3-D cues from MediaPipe's raw landmarks. z is depth on the same scale as x (smaller = nearer the
// camera), which is what separates pointing AT the camera (YOU) from pointing back at yourself (ME).
function geometry3(raw,w,h){
  const p=i=>[raw[i].x*w, raw[i].y*h, (raw[i].z||0)*w];
  const d=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1],a[2]-b[2]);
  const W=p(0);
  const F=[FING.index,FING.middle,FING.ring,FING.pinky];
  const ext=F.map(f=>d(p(f[3]),W) > d(p(f[1]),W)*1.1);
  const str=F.map(f=>{
    const chain=d(p(f[0]),p(f[1]))+d(p(f[1]),p(f[2]))+d(p(f[2]),p(f[3]));
    return chain? d(p(f[0]),p(f[3]))/chain : 0;
  });
  const folded=ext.filter(x=>!x).length, avgStr=(str[0]+str[1]+str[2]+str[3])/4;
  const a=p(5), b=p(8), len=d(a,b)||1;
  const point=ext[0] && !ext[1] && !ext[2] && !ext[3] && str[0]>0.8;
  const knuckles=[5,9,13,17].map(p);
  return {
    point,
    dir:(b[2]-a[2])/len,                  // -1 = index straight at the camera, +1 = straight back at the signer
    // straightness on the reference skeletons: flat B 1.0, curved C 0.90, O 0.75, fist S 0.27
    fist4: folded===4 && avgStr<0.55,
    bent: !point && avgStr>0.6 && avgStr<0.94,
    centre:[knuckles.reduce((s,k)=>s+k[0],0)/4, knuckles.reduce((s,k)=>s+k[1],0)/4]
  };
}

/* built-in signs (glosses). Each fires once per hold; the grammar below turns sequences into phrases. */
const POINT_DIR=0.45, GLOSS_GAP_MS=2600, GLOSS_COOLDOWN_MS=500;
// `shape` names one or two reference handshapes (keys into REF, the same 26 fingerspelling references used
// for the alphabet - see 00-constants.js) that stand in for this sign in the library's visual preview
// (07-library-ui.js). Most built-in signs are literally built from a fingerspelling handshape (HELLO is an
// open "B" hand waved, YES is an "S" fist nodded, and so on), so the same skeleton diagram doubles as this
// sign's glyph. ILY has no exact single-letter match (thumb+index+little finger extended isn't any letter
// on its own), so "Y" is used as the closest approximation - the written description still carries the
// precise handshape for that one.
const GLOSSES=[
  {id:"LOVE",  word:"Love",   hold:450, how:"Both fists crossed over your chest.", shape:["S","S"],
   test:c=>c.pair && c.pair.fists && c.pair.near<1.8},
  {id:"HOW",   word:"How?",   hold:400, how:"Both hands curved, knuckles together (start of the forward roll).", shape:["C","C"],
   test:c=>c.pair && c.pair.bent && c.pair.near<1.5},
  {id:"ILY",   word:"I love you.", hold:450, how:"Thumb, index and little finger up; middle and ring folded.", shape:["Y"],
   test:c=>c.g.ily},
  {id:"GOOD",  word:"Good.",  hold:450, how:"Thumbs-up: thumb up, the other four folded.", shape:["A"],
   test:c=>c.g.thumbsUp},
  {id:"YOU",   word:"You",    hold:350, how:"Index finger pointed straight at the camera.", shape:["D"],
   test:c=>c.g3.point && c.g3.dir<-POINT_DIR},
  {id:"ME",    word:"Me",     hold:350, how:"Index finger pointed back at your own chest.", shape:["D"],
   test:c=>c.g3.point && c.g3.dir>POINT_DIR},
  {id:"HELLO", word:"Hello.", hold:0,   how:"Open hand, waved side to side.", shape:["B"],
   test:c=>c.g.openPalm && c.m.x.changes>=2 && c.m.x.amp>0.55},
  {id:"YES",   word:"Yes.",   hold:0,   how:"Closed fist, nodded up and down.", shape:["S"],
   test:c=>c.g.fist && c.m.y.changes>=2 && c.m.y.amp>0.4}
];
const GLOSS_BY_ID=Object.fromEntries(GLOSSES.map(G=>[G.id,G]));
// ASL gloss order → English. The longest matching sequence wins.
const PHRASE_RULES=[
  {text:"I am good.",          g:["ME","GOOD"]},
  {text:"How are you?",        g:["HOW","YOU"]},
  {text:"Hello, how are you?", g:["HELLO","HOW","YOU"]},
  {text:"Love you.",           g:["LOVE","YOU"]},
  {text:"I love you.",         g:["ME","LOVE","YOU"]},
  {text:"I love you.",         g:["ILY"]},
  {text:"You good?",           g:["YOU","GOOD"]}
];
let glossState={}, glossLastFire=0, lastPointDir=null;
let glossQ=[], glossDeadline=0, glossDone=null;
