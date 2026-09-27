/* ============================================================
   10 · phrase recording
   ============================================================ */
let recState=null;   // {id, start}
function countdown(n,done){
  const el=els("bigcount");
  el.className="bigcount on"; el.textContent=n;
  const tick=()=>{
    n--;
    if(n<=0){ el.className="bigcount"; done(); return; }
    el.textContent=n; setTimeout(tick,700);
  };
  setTimeout(tick,700);
}
function startRecording(id){
  if(!session){ alert("Start the camera first."); return; }
  const slot=P().slots.find(s=>s.id===id);
  if(!slot){ alert("That phrase slot doesn't exist any more — refresh the library and try again."); return; }
  countdown(3,()=>{
    recState={id,start:performance.now()};
    const el=els("bigcount");
    el.className="bigcount rec on";
    el.textContent="● Recording “"+slot.text+"”";
    setTimeout(()=>{
      el.className="bigcount";
      finishRecording();
    },PH_REC_MS);
  });
}
function finishRecording(){
  if(!recState) return;
  const {id,start}=recState; recState=null;
  const end=performance.now();
  const win=buf.filter(f=>f.t>=start&&f.t<=end);
  if(win.length<6){ alert("Not enough hand frames in that take — keep a hand in view and try again."); return; }
  const rs=resampleFrames(win,PH_T), V=windowVec(rs);
  if(!V){ alert("No hand was detected during that take."); return; }
  const prof=P();
  prof.takes[id]=prof.takes[id]||[];
  prof.takes[id].push({dur:Math.round(end-start), f:V.map(v=>Array.from(v).map(x=>+x.toFixed(3)))});
  if(prof.takes[id].length>PH_TAKES) prof.takes[id].shift();
  save(); buildPhraseLists(); refreshProfileUI();
}
