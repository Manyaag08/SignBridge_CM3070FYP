/* ============================================================
   5 · phrase trajectory engine
   ============================================================ */
let buf=[];                                  // rolling [{t, slots:[handOrNull, handOrNull]}]
function pushFrame(hands,t){
  buf.push({t, slots:[hands[0]||null, hands[1]||null]});
  const cut=t-PH_BUF_MS;
  while(buf.length && buf[0].t<cut) buf.shift();
}
function resampleFrames(frames,T){
  if(!frames.length) return null;
  const t0=frames[0].t, t1=frames[frames.length-1].t, out=[];
  for(let i=0;i<T;i++){
    const tt=t0+(t1-t0)*(T===1?0:i/(T-1));
    let bi=0,bd=Infinity;
    for(let j=0;j<frames.length;j++){
      const dd=Math.abs(frames[j].t-tt);
      if(dd<bd){bd=dd;bi=j;}
    }
    out.push(frames[bi]);
  }
  return out;
}
// 90-d per frame: [slot0 shape 42 | slot0 wrist dx,dy | slot1 shape 42 | slot1 wrist dx,dy | presence 2]
function windowVec(frames){
  const ref=frames.find(f=>f.slots[0]);
  if(!ref) return null;
  const ox=ref.slots[0].wrist[0], oy=ref.slots[0].wrist[1];
  let sc=0,n=0;
  for(const f of frames) for(const s of f.slots) if(s){ sc+=s.scale; n++; }
  sc = n ? sc/n : 1; if(sc<1) sc=1;
  return frames.map(f=>{
    const v=new Float32Array(90);
    for(let k=0;k<2;k++){
      const s=f.slots[k], o=k*44;
      if(!s) continue;
      v[88+k]=1;
      for(let j=0;j<42;j++) v[o+j]=s.feat[j];
      v[o+42]=(s.wrist[0]-ox)/sc;
      v[o+43]=(s.wrist[1]-oy)/sc;
    }
    return v;
  });
}
function seqDist(A,B){
  let tot=0;
  for(let i=0;i<A.length;i++){
    const a=A[i], b=B[i];
    let cost=0, active=0;
    for(let k=0;k<2;k++){
      const pa=a[88+k], pb=b[88+k];
      if(!pa && !pb) continue;
      active++;
      if(pa!==pb){ cost+=0.8; continue; }
      const o=k*44;
      let sd=0;
      for(let j=0;j<42;j++) sd+=Math.abs(a[o+j]-b[o+j]);
      sd/=42;
      const dx=a[o+42]-b[o+42], dy=a[o+43]-b[o+43];
      const pd=Math.min(Math.hypot(dx,dy)/2,1);
      cost += 0.55*Math.min(sd/0.5,1) + 0.45*pd;
    }
    tot += active ? cost/active : 0;
  }
  return tot/A.length;
}
function bestPhraseMatch(now){
  const takes=P().takes, ids=Object.keys(takes);
  if(!ids.length) return null;
  let best=null;
  for(const id of ids){
    for(const tk of takes[id]){
      const win=[];
      for(let i=buf.length-1;i>=0;i--){ if(buf[i].t < now-tk.dur) break; win.unshift(buf[i]); }
      if(win.length<6) continue;
      const rs=resampleFrames(win,PH_T);
      const V=windowVec(rs);
      if(!V) continue;
      const d=seqDist(V,takeVecs(tk));
      if(!best||d<best.d) best={d,id};
    }
  }
  return best;
}
