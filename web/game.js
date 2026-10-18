// Same physics as dino/config.py + dino/physics.py + dino/env.py
const C={W:600,H:200,DX:50,DW:30,DH:40,G:0.6,V0:12,S0:5,HMIN:20,HMAX:70,WMIN:14,WMAX:30,MARGIN:30,EXTRA:150,GROUND:30};
const apex=C.V0*C.V0/(2*C.G), T=2*C.V0/C.G;
const clearT=h=>{const r=Math.sqrt(C.V0*C.V0-2*C.G*h);return[(C.V0-r)/C.G,(C.V0+r)/C.G]};
const dStar=(s,h,w)=>s*clearT(h)[1]-w-C.DW;
const minSpace=(s,a,b,w)=>s*(clearT(a)[0]+1)+C.DW+dStar(s,b,w)+C.MARGIN;
const ri=(a,b)=>a+Math.floor(Math.random()*(b-a+1));
// Sprites: put dino.png, tree.png, cloud.png and ground.png (transparent background) next to index.html.
// If a file is missing the game falls back to plain rectangles.
function loadImg(src){const i=new Image();i.ready=false;i.onload=()=>i.ready=true;i.src=src;return i}
const IMG={dino:loadImg("dino.png"),tree:loadImg("tree.png"),cloud:loadImg("cloud.png"),ground:loadImg("ground.png")};
function sprite(img,x,y,w,h,fg){ // y = top of the hitbox; image is stretched to the hitbox
 if(img.ready)ctx.drawImage(img,x,y,w,h);else{ctx.fillStyle=fg;ctx.fillRect(x,y,w,h)}}
//CLOUDS-BEGIN  (decorative only: does not touch the physics or the training)
const CLOUD_H=[18,30], CLOUD_PAD=14, CLOUD_MAX=5, TOP_THIRD=C.H/3;
let clouds=[];
const cAspect=()=>IMG.cloud.ready?IMG.cloud.naturalWidth/IMG.cloud.naturalHeight:2.4;
const cw=c=>c.h*cAspect();
const hit=(a,b)=>a.x<b.x+cw(b)+CLOUD_PAD&&b.x<a.x+cw(a)+CLOUD_PAD&&a.y<b.y+b.h+CLOUD_PAD&&b.y<a.y+a.h+CLOUD_PAD;
function tryAddCloud(x0,x1){for(let k=0;k<12;k++){const h=CLOUD_H[0]+Math.random()*(CLOUD_H[1]-CLOUD_H[0]);
 const c={x:x0+Math.random()*(x1-x0),y:Math.random()*(TOP_THIRD-h),h};
 if(!clouds.some(o=>hit(c,o))){clouds.push(c);return true}}return false}
function initClouds(){clouds=[];for(let i=0;i<CLOUD_MAX;i++)tryAddCloud(0,C.W)}
function updateClouds(speed){const v=Math.max(0.3,speed*0.15);clouds.forEach(c=>c.x-=v);
 clouds=clouds.filter(c=>c.x+cw(c)>0);if(clouds.length<CLOUD_MAX&&Math.random()<0.03)tryAddCloud(C.W,C.W+120)}
function drawClouds(fg){clouds.forEach(c=>{if(IMG.cloud.ready)ctx.drawImage(IMG.cloud,c.x,c.y,cw(c),c.h);
 else{ctx.save();ctx.globalAlpha=0.2;ctx.fillStyle=fg;ctx.beginPath();ctx.ellipse(c.x+cw(c)/2,c.y+c.h/2,cw(c)/2,c.h/2,0,0,7);ctx.fill();ctx.restore()}})}
//CLOUDS-END
// Ground: constant (not scrolling). Set GROUND_SCROLL=true to make it move with the speed.
const GROUND_SCROLL=false;
function drawGround(gy,fg){const im=IMG.ground;
 if(!im.ready){ctx.fillStyle=fg;ctx.fillRect(0,gy,C.W,2);return}
 const h=C.GROUND,w=Math.max(1,Math.round(im.naturalWidth*h/im.naturalHeight));
 const off=GROUND_SCROLL?-((g.frame*g.s)%w):0;for(let x=off;x<C.W;x+=w)ctx.drawImage(im,x,gy,w,h)}
let g,mode="human",jumpQ=false,net=null,best=0;
function reset(){g={tAir:null,y:0,s:C.S0,frame:0,score:0,trees:[],dead:false};fill();initClouds();}
function spawn(){const h=ri(C.HMIN,C.HMAX),w=ri(C.WMIN,C.WMAX);let x;
 if(!g.trees.length)x=C.W+100;else{const t=g.trees[g.trees.length-1];
  x=t.x+t.w+minSpace(g.s*1.1,t.h,h,w)+Math.random()*C.EXTRA}
 g.trees.push({x,w,h,p:false})}
function fill(){while(ahead().length<4)spawn()}
const ahead=()=>g.trees.filter(t=>t.x+t.w>C.DX);
function obs(){const a=ahead().slice(0,2),f=[];
 for(const t of a){const d=t.x-C.DX-C.DW;f.push(d/C.W,d/(g.s*T),t.h/C.HMAX,t.w/C.WMAX)}
 while(f.length<8)f.push(0);
 f.push(g.s/16,g.y/apex,g.tAir===null?0:g.tAir/T);return f}
function q(x){net.layers.forEach((L,i)=>{x=L.W.map((row,j)=>row.reduce((s,v,k)=>s+v*x[k],L.b[j]));
 if(i<net.layers.length-1)x=x.map(v=>Math.max(0,v))});return x}
function decide(){
 if(g.tAir!==null)return 0;
 if(mode==="human"){const j=jumpQ;jumpQ=false;return j?1:0}
 if(mode==="analytic"){const t=ahead()[0];return t&&(t.x-C.DX-C.DW)<=dStar(g.s,t.h,t.w)-2?1:0}
 if(mode==="ai"&&net){const o=q(obs());return o[1]>o[0]?1:0}
 return 0}
function step(){
 if(g.dead)return;
 if(decide()===1&&g.tAir===null)g.tAir=0;
 if(g.tAir!==null){g.tAir++;const t=g.tAir,y=C.V0*t-0.5*C.G*t*t;if(y<=0){g.y=0;g.tAir=null}else g.y=y}
 g.trees.forEach(t=>t.x-=g.s);g.frame++;
 if(g.trees.some(t=>t.x<C.DX+C.DW&&t.x+t.w>C.DX&&g.y<t.h))g.dead=true;
 g.trees.forEach(t=>{if(!t.p&&t.x+t.w<C.DX){t.p=true;g.score++;if(g.score%10===0)g.s*=1.1}});
 g.trees=g.trees.filter(t=>t.x+t.w>-50);fill();best=Math.max(best,g.score);updateClouds(g.s)}
const cv=document.getElementById("c"),ctx=cv.getContext("2d"),info=document.getElementById("info");
function draw(){const dark=matchMedia("(prefers-color-scheme:dark)").matches,fg=dark?"#ddd":"#333";
 ctx.clearRect(0,0,C.W,C.H);ctx.fillStyle=fg;const gy=C.H-C.GROUND;drawClouds(fg);drawGround(gy,fg);
 sprite(IMG.dino,C.DX,gy-g.y-C.DH,C.DW,C.DH,fg);
 g.trees.forEach(t=>sprite(IMG.tree,t.x,gy-t.h,t.w,t.h,fg));
 if(g.dead){ctx.font="20px sans-serif";ctx.fillText("Game over - press space",200,80)}
 info.textContent=`mode: ${mode}${mode==="ai"&&!net?" (no weights.json found)":""} | score ${g.score} | best ${best} | speed ${g.s.toFixed(1)} px/frame`}
let last=0,acc=0;
function loop(ts){acc+=Math.min(100,ts-last);last=ts;while(acc>=1000/60){step();acc-=1000/60}draw();requestAnimationFrame(loop)}
addEventListener("keydown",e=>{if(e.code==="Space"||e.code==="ArrowUp"){e.preventDefault();if(g.dead)reset();else jumpQ=true}});
cv.addEventListener("pointerdown",()=>{if(g.dead)reset();else jumpQ=true});
document.querySelectorAll("button").forEach(b=>b.onclick=()=>{mode=b.dataset.m;reset()});
fetch("weights.json").then(r=>r.json()).then(j=>net=j).catch(()=>{});
reset();requestAnimationFrame(loop);
