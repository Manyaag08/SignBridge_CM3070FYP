/* ============================================================
   14 · signing avatar: text → ASL sign plan → animated skeleton
   ============================================================ */
// Must stay in step with plan_avatar() in evaluation/signbridge_core.py (a test checks this).
const AVATAR_LEXICON={hello:"HELLO",hi:"HELLO",how:"HOW",you:"YOU",your:"YOU",i:"ME",me:"ME",my:"ME",
  good:"GOOD",fine:"GOOD",love:"LOVE",yes:"YES",thank:"THANK-YOU",thanks:"THANK-YOU",please:"PLEASE",
  sorry:"SORRY",help:"HELP",name:"NAME",no:"NO",what:"WHAT",where:"WHERE",need:"NEED",feel:"FEEL"};
const AVATAR_DROP=new Set(["am","is","are","a","an","the","to","do","does","be"]);
const SIGN_S=0.9, LETTER_S=0.4, PAUSE_S=0.25;
function planAvatar(text){
  const words=(text.toLowerCase().match(/[a-z']+/g)||[]), plan=[];
  for(let i=0;i<words.length;i++){
    let w=words[i];
    if(w==="thank" && words[i+1]==="you"){ plan.push({type:"sign",value:"THANK-YOU"}); i++; continue; }
    if(AVATAR_DROP.has(w)) continue;
    w=w.replace(/'/g,"");
    if(AVATAR_LEXICON[w]) plan.push({type:"sign",value:AVATAR_LEXICON[w]});
    else { const v=w.replace(/[^a-z]/g,"").toUpperCase(); if(v) plan.push({type:"spell",value:v}); }
  }
  return plan;
}
// Sign keyframes. Positions are in body units (0,0 = chest centre, y up is negative, chin ≈ -0.55).
// shape = a reference handshape; path animates the anchor over the sign's duration.
const SIGNS={
  "HELLO":    [{shape:"B",from:[0.35,-0.8],to:[0.6,-0.72],path:"wave"}],
  "HOW":      [{shape:"C",from:[-0.1,0],to:[-0.1,-0.05],rot:[0,70]},{shape:"C",from:[0.1,0],to:[0.1,-0.05],rot:[0,-70],mirror:true}],
  "YOU":      [{shape:"D",from:[0.12,-0.05],to:[0.28,0.02],scale:[1,1.35],rot:[-20,-20]}],
  "ME":       [{shape:"D",from:[0.15,-0.05],to:[0.05,-0.12],rot:[160,160],path:"tap"}],
  "GOOD":     [{shape:"B",from:[0.05,-0.55],to:[0.18,0.02]}],
  "LOVE":     [{shape:"S",from:[-0.08,-0.1],to:[-0.04,-0.12],rot:[35,35]},{shape:"S",from:[0.08,-0.1],to:[0.04,-0.12],rot:[-35,-35],mirror:true}],
  "YES":      [{shape:"S",from:[0.28,-0.1],to:[0.28,-0.1],path:"nod"}],
  "THANK-YOU":[{shape:"B",from:[0.05,-0.58],to:[0.35,-0.2]}],
  "PLEASE":   [{shape:"B",from:[0.05,-0.1],to:[0.05,-0.1],path:"circle"}],
  "SORRY":    [{shape:"A",from:[0.05,-0.1],to:[0.05,-0.1],path:"circle"}],
  "HELP":     [{shape:"A",from:[0.12,0.2],to:[0.12,-0.02]},{shape:"B",from:[0.12,0.32],to:[0.12,0.1],rot:[-90,-90],mirror:true}],
  "NAME":     [{shape:"H",from:[0.0,-0.08],to:[0.0,-0.08],rot:[-60,-60],path:"tap"},{shape:"H",from:[0.14,-0.14],to:[0.14,-0.14],rot:[-30,-30],mirror:true}],
  "NO":       [{shape:"U",from:[0.28,-0.2],to:[0.28,-0.2],path:"tap"}],
  "WHAT":     [{shape:"B",from:[-0.28,0.1],to:[-0.28,0.1],rot:[80,80],path:"wave"},{shape:"B",from:[0.28,0.1],to:[0.28,0.1],rot:[-80,-80],path:"wave",mirror:true}],
  "WHERE":    [{shape:"D",from:[0.32,-0.35],to:[0.32,-0.35],path:"wave"}],
  "NEED":     [{shape:"X",from:[0.25,-0.05],to:[0.25,0.05],path:"nod"}],
  "FEEL":     [{shape:"B",from:[0.05,0.05],to:[0.05,-0.22]}]
};
const av={queue:[], item:null, t0:0, plan:[], idx:-1};
const avCanvas=els("avatar"), avCtx=avCanvas.getContext("2d");
function avSpeed(){ return (+els("avSpeed").value)/100; }
function avatarSay(text,plan){
  plan=plan||planAvatar(text);   // a ready-made plan lets practice mode fingerspell a single letter
  if(!plan.length) return;
  // flatten: a spelled word becomes one step per letter so the plan shows progress letter by letter
  const steps=[];
  plan.forEach((p,pi)=>{
    if(p.type==="sign") steps.push({kind:"sign",value:p.value,dur:SIGN_S,plan:pi});
    else [...p.value].forEach(ch=>{ if(REF[ch]) steps.push({kind:"letter",value:ch,dur:LETTER_S,plan:pi}); });
    steps.push({kind:"pause",dur:PAUSE_S,plan:pi});
  });
  av.queue.push({text,plan,steps});
  if(!av.item) avNext();
}
function avNext(){
  const job=av.queue.shift();
  if(!job){ av.item=null; els("avStatus").innerHTML='<span class="dot"></span>Ready'; els("avStatus").className="status"; return; }
  av.item=job; av.idx=0; av.t0=performance.now();
  els("avplan").innerHTML=job.plan.map((p,i)=>'<span class="'+(p.type==="sign"?"sign":"")+'" data-i="'+i+'">'+
    (p.type==="sign"?p.value:"#"+p.value)+'</span>').join("");
  const total=job.steps.reduce((a,s)=>a+s.dur,0);
  els("avStatus").innerHTML='<span class="dot"></span>Signing “'+job.text.replace(/</g,"&lt;")+'” · '+(total/avSpeed()).toFixed(1)+' s';
  els("avStatus").className="status ok";
}
els("avBtn").onclick=()=>{ const t=els("avText").value.trim(); if(t){ avatarSay(t); els("avText").value=""; } };
els("avText").onkeydown=e=>{ if(e.key==="Enter") els("avBtn").click(); };

function drawHand(ctx,shape,x,y,size,rotDeg,mirror,alpha){
  const pts=REF[shape]; if(!pts) return;
  const r=rotDeg*Math.PI/180, c=Math.cos(r), s=Math.sin(r), m=mirror?-1:1;
  const P=pts.map(p=>{ const px=p[0]*m, py=p[1]; return [x+(px*c-py*s)*size, y+(px*s+py*c)*size]; });
  ctx.globalAlpha=alpha;
  ctx.strokeStyle="#2e8b6f"; ctx.lineWidth=2.4; ctx.lineCap="round";
  for(const [a,b] of CONNS){ ctx.beginPath(); ctx.moveTo(P[a][0],P[a][1]); ctx.lineTo(P[b][0],P[b][1]); ctx.stroke(); }
  ctx.fillStyle="#5eead4";
  for(const p of P){ ctx.beginPath(); ctx.arc(p[0],p[1],2.1,0,Math.PI*2); ctx.fill(); }
  ctx.globalAlpha=1;
}
function drawBody(ctx,W,H){
  const cx=W/2, cy=H*0.62, u=H*0.5;
  ctx.fillStyle="#16201d"; ctx.strokeStyle="#1f2b28"; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(cx-u*0.62,H); ctx.quadraticCurveTo(cx-u*0.6,cy-u*0.28,cx,cy-u*0.3);
  ctx.quadraticCurveTo(cx+u*0.6,cy-u*0.28,cx+u*0.62,H); ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.ellipse(cx,cy-u*0.62,u*0.2,u*0.25,0,0,Math.PI*2); ctx.fill(); ctx.stroke();
  return {cx,cy,u};
}
function pathOffset(path,k){
  if(path==="wave") return [Math.sin(k*Math.PI*4)*0.06,0];
  if(path==="nod") return [0,Math.abs(Math.sin(k*Math.PI*3))*0.08];
  if(path==="tap") return [0,Math.abs(Math.sin(k*Math.PI*2))*-0.05];
  if(path==="circle") return [Math.cos(k*Math.PI*2)*0.07,Math.sin(k*Math.PI*2)*0.07];
  return [0,0];
}
function avatarFrame(){
  const W=avCanvas.width, H=avCanvas.height;
  avCtx.clearRect(0,0,W,H);
  const {cx,cy,u}=drawBody(avCtx,W,H);
  const toXY=(p)=>[cx+p[0]*u, cy+p[1]*u];
  let label="";
  if(av.item){
    const job=av.item, sp=avSpeed();
    let el=(performance.now()-av.t0)/1000*sp, i=0;
    while(i<job.steps.length && el>job.steps[i].dur){ el-=job.steps[i].dur; i++; }
    if(i>=job.steps.length){ avNext(); }
    else{
      const st=job.steps[i], k=clamp(el/st.dur,0,1);
      els("avplan").querySelectorAll("span").forEach(sp=>sp.classList.toggle("now",+sp.dataset.i===st.plan));
      if(st.kind==="sign"){
        label=st.value;
        for(const h of SIGNS[st.value]||[]){
          const e=k<0.5?2*k*k:1-Math.pow(-2*k+2,2)/2, off=pathOffset(h.path,k);
          const pos=[h.from[0]+(h.to[0]-h.from[0])*e+off[0], h.from[1]+(h.to[1]-h.from[1])*e+off[1]];
          const rot=h.rot?h.rot[0]+(h.rot[1]-h.rot[0])*e:0, sc=h.scale?h.scale[0]+(h.scale[1]-h.scale[0])*e:1;
          const [x,y]=toXY(pos);
          drawHand(avCtx,h.shape,x,y,u*0.34*sc,rot,!!h.mirror,1);
        }
      }else if(st.kind==="letter"){
        label="#"+st.value;
        let off=[0,0];
        if(st.value==="J") off=[Math.sin(k*Math.PI)*-0.08,k*0.1];
        if(st.value==="Z") off=[[0,0.12,0,0.12][Math.min(3,Math.floor(k*4))]-0.06,Math.floor(k*3)*0.05];
        const [x,y]=toXY([0.3+off[0],-0.35+off[1]]);
        drawHand(avCtx,st.value,x,y,u*0.42,0,false,1);
      }
    }
  }
  avCtx.font="600 18px ui-monospace, Menlo, monospace"; avCtx.fillStyle="#5eead4"; avCtx.textAlign="left";
  avCtx.fillText(label,12,26);
  requestAnimationFrame(avatarFrame);
}
requestAnimationFrame(avatarFrame);
