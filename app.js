import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
const TARGET_PHRASE="يساعدنا الغذاء الصحي وممارسة الرياضة على النمو السليم والنجاح في دراستنا";
const MAX_RECORDING_TIME=10000,SILENCE_DURATION=3000,SILENCE_THRESHOLD=0.015,WHISPER_SAMPLE_RATE=16000;
let transcriber=null,busy=false;
const charger=document.getElementById("charger"),parler=document.getElementById("parler"),etat=document.getElementById("etat"),resultat=document.getElementById("resultat"),comparaison=document.getElementById("comparaison");

charger.onclick=async()=>{
 charger.disabled=true; etat.textContent="Chargement de Whisper...";
 try{
  transcriber=await pipeline("automatic-speech-recognition","onnx-community/whisper-small");
  etat.textContent="Whisper est prêt."; parler.disabled=false;
 }catch(e){console.error(e);etat.textContent="Erreur de chargement. Ouvre F12 > Console.";charger.disabled=false;}
};

parler.onclick=async()=>{
 if(!transcriber||busy)return;
 busy=true;parler.disabled=true;resultat.textContent="—";comparaison.textContent="—";
 try{
  const audio=await recordAudio();
  if(!audio.length){etat.textContent="Aucune parole détectée.";return;}
  etat.textContent="Analyse de ta réponse...";
  const output=await transcriber(audio,{language:"ar",task:"transcribe"});
  const recognized=(output?.text||"").trim();
  resultat.textContent=recognized||"(aucun texte reconnu)";
  const a=normalizeArabic(recognized).replace(/\s/g,"");
  const b=normalizeArabic(TARGET_PHRASE).replace(/\s/g,"");
  const distance=levenshtein(a,b);
  const pct=b.length?Math.round(distance/b.length*100):100;
  comparaison.textContent=`Distance : ${distance} — Erreur : ${pct}% — ${distance<=2?"BRAVO":"ESSAYE ENCORE"}`;
  etat.textContent="Analyse terminée.";
 }catch(e){console.error(e);etat.textContent="Erreur. Ouvre F12 > Console.";}
 finally{busy=false;parler.disabled=!transcriber;}
};

async function recordAudio(){
 const stream=await navigator.mediaDevices.getUserMedia({audio:true});
 const AC=window.AudioContext||window.webkitAudioContext,ctx=new AC(); await ctx.resume();
 const src=ctx.createMediaStreamSource(stream),proc=ctx.createScriptProcessor(4096,1,1),chunks=[];
 let spoke=false,lastVoice=Date.now(),done=false,timer;
 etat.textContent="Parle maintenant...";
 return new Promise(resolve=>{
  const finish=()=>{
   if(done)return;done=true;clearTimeout(timer);
   try{proc.disconnect();src.disconnect();}catch(_){}
   stream.getTracks().forEach(t=>t.stop());
   const merged=merge(chunks),rate=ctx.sampleRate;ctx.close().catch(()=>{});
   resolve(spoke&&merged.length?resample(merged,rate,16000):new Float32Array(0));
  };
  proc.onaudioprocess=e=>{
   if(done)return;
   const x=e.inputBuffer.getChannelData(0);chunks.push(new Float32Array(x));
   let s=0;for(let i=0;i<x.length;i++)s+=x[i]*x[i];
   const rms=Math.sqrt(s/x.length),now=Date.now();
   if(rms>SILENCE_THRESHOLD){spoke=true;lastVoice=now;}
   if(spoke&&now-lastVoice>=SILENCE_DURATION)finish();
  };
  src.connect(proc);proc.connect(ctx.destination);timer=setTimeout(finish,MAX_RECORDING_TIME);
 });
}
function merge(chunks){let n=chunks.reduce((s,x)=>s+x.length,0),o=new Float32Array(n),p=0;for(const x of chunks){o.set(x,p);p+=x.length;}return o;}
function resample(x,from,to){if(from===to)return x;const r=from/to,n=Math.round(x.length/r),o=new Float32Array(n);for(let i=0;i<n;i++){const pos=i*r,j=Math.floor(pos),f=pos-j,a=x[j]??0,b=x[Math.min(j+1,x.length-1)]??a;o[i]=a+(b-a)*f;}return o;}
function normalizeArabic(t){return t.replace(/[ًٌٍَُِّْـ]/g,"").replace(/[آأإٱ]/g,"ا").replace(/\s+/g," ").replace(/ة/g,"ه").replace(/[،؛؟.!?,:]/g,"").replace(/[\u200B-\u200D\uFEFF]/g,"").replace(/[^\u0600-\u06FF\s]/g,"").trim();}
function levenshtein(a,b){const m=Array.from({length:b.length+1},()=>Array(a.length+1));for(let i=0;i<=b.length;i++)m[i][0]=i;for(let j=0;j<=a.length;j++)m[0][j]=j;for(let i=1;i<=b.length;i++)for(let j=1;j<=a.length;j++)m[i][j]=Math.min(m[i-1][j]+1,m[i][j-1]+1,m[i-1][j-1]+(b[i-1]===a[j-1]?0:1));return m[b.length][a.length];}
