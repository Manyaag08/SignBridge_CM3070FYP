/* ============================================================
   13 · recognition model selector
   ============================================================ */
const MODEL_PATHS={multi:"./artifacts/asl_landmark_mlp_multisigner_web.onnx", single:"./artifacts/asl_landmark_mlp_web.onnx"};
try{ const m=localStorage.getItem("signbridge.model"); if(m&&MODEL_PATHS[m]) els("modelSel").value=m; }catch(e){}
els("modelSel").onchange=async e=>{
  try{ localStorage.setItem("signbridge.model",e.target.value); }catch(err){}
  els("modelInfo").textContent=e.target.value==="multi"
    ? "Recommended — trained on many signers for the best accuracy on new hands."
    : "The original model, trained on a single signer.";
  if(session){
    try{ session=await ort.InferenceSession.create(MODEL_PATHS[e.target.value]); setStatus("Recognition model switched",true); }
    catch(err){ setStatus("Couldn't load that model","err"); }
  }
};
