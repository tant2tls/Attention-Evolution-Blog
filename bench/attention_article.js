(()=>{
'use strict';
const D=JSON.parse(document.getElementById('h100-study-data').textContent),M=D.models;
const $=id=>document.getElementById(id),esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(x,n=2)=>Number(x).toLocaleString('en-US',{minimumFractionDigits:n,maximumFractionDigits:n});
const context=c=>({1024:'1K',16384:'16K',65536:'64K',131072:'128K'})[c];
const categories=['ordinary_kv','window_kv','shared_compressed','sparse_index_tail','recurrent'];
const labels=['KV / latents','Window KV','Shared / compressed KV','Index + tail','Recurrent state'];
const colors=['#5076a6','#b7783d','#ba6b85','#7b6aa6','#228b8b'];
const colorType=t=>t==='Recurrent'?'#228b8b':t==='Full GQA'?'#5076a6':t==='Window'?'#b7783d':t.includes('compressed')?'#ba6b85':'#7b6aa6';
const report='report/'+D.study+'/';
let selected=M[0].id,layerIndex=0;
const stories={
'llama-31-70b':['Start with a cache you can count.','All 80 layers retain full-history K/V. Sixty-four query heads share eight KV heads, reducing head duplication while keeping every earlier token accessible.','The input grows, and so does every layer’s cache. GQA bounds neither the context length nor the number of growing layers.'],
'mimo-v26':['Let most layers look nearby.','Thirty-nine layers retain a 128-token window; nine global GQA layers retain the full history. Their positions come from the checkpoint, including a global first layer.','Local storage stops growing, but the global layers still carry the distant past. This deployment uses BF16 KV after the FP8 path failed validation.'],
'v4-0731':['Compress the distant past.','Two layers are window-only. Twenty-one combine a local window with fourfold-compressed history and an indexer; twenty combine a window with 128-fold-compressed history.','A shorter history representation also requires compression, selection and temporary state. Fewer entries are not a measured speedup by themselves.'],
'qwen-38-fp8':['Keep a running state—and a retrieval path.','Thirty-six Gated DeltaNet layers update recurrent state. Twelve sparse QSA layers retain history and select entries through an indexer. We use the official FP8-weight checkpoint.','Only the recurrent portion has bounded state. Sparse K/V, compressed index keys and ring buffers must be counted too; their dtypes need not match the FP8 model name.'],
'glm-53':['Combine recurrent state with sparse latents.','Thirty-four Kimi Delta Attention layers maintain recurrent state, while eleven sparse MLA layers retrieve from compressed KV latents with index and tail buffers.','A small decode cost does not promise a small prefill cost. This deployment completed the grid after an intermittent failure and a retry under the same frozen settings.'],
'v41':['Let another layer own the history.','Forty layers use windows alongside sparse or compressed history. Four configured KV source layers provide buffers that later layers can read. Index state has its own sharing map.','Readers do not create another copy of the shared long-range KV. They can still own local windows. Exact consumer links shown here are inferred from config, not directly traced.']};
function table(headers,rows){return '<table class="value-table"><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map((v,i)=>(i?'<td>':'<th>')+v+(i?'</td>':'</th>')).join('')+'</tr>').join('')+'</tbody></table>'}
function legend(id,items){$(id).innerHTML=items.map(([name,color])=>'<span><i class="swatch" style="background:'+color+'"></i>'+esc(name)+'</span>').join('')}
function barRows(id,rows,max,unit,decimals=2){
 $(id).innerHTML=rows.map(r=>'<div class="bar-row"><span>'+esc(r.name)+'</span><div class="bar-track">'+(r.value===null?'':('<div class="bar-fill" style="width:'+Math.max(.15,100*r.value/max)+'%;background:'+r.color+'"></div>')+(r.min!==undefined?'<span class="whisker" style="left:'+100*r.min/max+'%;width:'+Math.max(.2,100*(r.max-r.min)/max)+'%"></span>':''))+'</div><span class="bar-value">'+(r.value===null?'unsupported':num(r.value,decimals)+' '+unit)+'</span></div>').join('');
}
function renderArchitecture(){
 $('model-tabs').innerHTML=M.map(m=>'<button type="button" data-model="'+m.id+'" aria-pressed="'+(m.id===selected)+'">'+esc(m.name)+'</button>').join('');
 const m=M.find(x=>x.id===selected),st=stories[m.id];
 $('architecture-image').src='assets/attention/'+m.id+'.svg';$('architecture-image').alt=m.name+': '+st[1];
 $('architecture-sequence').innerHTML=m.layer_data.map(l=>'<button type="button" data-architecture-layer="'+l.index+'" title="Layer '+l.index+': '+esc(l.type)+'" style="background:'+colorType(l.type)+'" aria-label="Layer '+l.index+': '+esc(l.type)+'">'+l.index+'</button>').join('');
 legend('architecture-legend',[...new Set(m.layer_data.map(l=>l.type))].map(t=>[t,colorType(t)]));
 $('architecture-story').innerHTML='<span class="badge">'+esc(m.kind)+' · '+m.layers+' layers</span><h3>'+esc(st[0])+'</h3><p>'+esc(st[1])+'</p><p>'+esc(st[2])+'</p><a class="tiny" href="https://huggingface.co/'+m.repo+'/blob/'+m.revision+'/config.json" target="_blank" rel="noopener">Checkpoint config at '+m.revision.slice(0,12)+' ↗</a>';
}
function renderParameters(){
 barRows('parameter-total',M.map(m=>({name:m.name,value:m.total,color:m.color})),Math.max(...M.map(m=>m.total)),'B',1);
 barRows('parameter-active',M.map(m=>({name:m.name,value:m.active,color:m.color})),Math.max(...M.map(m=>m.active)),'B',2);
 $('architecture-table').innerHTML=table(['Deployment','Layers','Attention design','Total B','Lookup B','Auxiliary/MTP B','Active estimate B'],M.map(m=>[esc(m.name),m.layers,esc(m.kind),num(m.total),num(m.lookup),num(m.mtp),num(m.active)]));
}
const storageOrder=['ordinary_kv','window_kv','shared_compressed','sparse_index_tail','recurrent_matrix','sconv','state_padding','recurrent'];
const storagePalette={ordinary_kv:'#91b4ed',window_kv:'#a2cdbb',shared_compressed:'#c995ad',sparse_index_tail:'#aa9acb',recurrent_matrix:'#5daba6',sconv:'#dbb574',state_padding:'#b8bdb9',recurrent:'#5daba6'};
let storageCategory='ordinary_kv';
function storageLabel(m,key){
 if(key==='ordinary_kv')return m.id==='qwen-38-fp8'?'sparse_attn':m.id==='glm-53'?'sparse_mla':'global_attn';
 if(key==='shared_compressed')return m.id==='v41'?'shared_kv':'compressed_kv';
 return {window_kv:'swa_attn',sparse_index_tail:'index_tail',recurrent_matrix:'recurrent_state',sconv:'sconv',state_padding:'state_padding',recurrent:'recurrent_state'}[key];
}
const storageDescription={ordinary_kv:'K/V or latent state retained by the full-history or sparse-attention layers. Sparse retrieval can still retain a growing history.',window_kv:'K/V for recent sliding-window attention. Old entries can expire after the window has moved on.',shared_compressed:'Long-range state in compressed entries or buffers reused by other layers. Shared buffers are counted at their owners.',sparse_index_tail:'Indexer keys and auxiliary ring/tail or compressor state required by the deployed attention path.',recurrent_matrix:'The recurrent matrix updated during generation. Its shape is fixed; its precision and size come from the runtime specification.',sconv:'Short-convolution history used by the recurrent layers. Bytes are calculated from recorded tensor shapes/dtypes and observed live pages.',state_padding:'Unused bytes inside occupied recurrent-state pages, derived from the recorded page size minus the short-conv and matrix payloads.',recurrent:'The occupied state of recurrent-attention layers, including their auxiliary buffers.'};
function storageSelection(){const m=M.find(m=>m.id===$('storage-model').value),c=$('memory-context').value;return {m,c,d:m.memory[c],perGpu:$('storage-scope').value==='gpu'}}
function storageAmount(gib,perGpu){return perGpu?num(gib*1024/8,1)+' MiB':num(gib,3)+' GiB'}
function storagePercent(value,total){const p=100*value/total;return p>0&&p<.1?'<0.1%':num(p,1)+'%'}
function renderStorage(){
 const {m,c,d,perGpu}=storageSelection();$('storage-context-value').textContent=num(+c,0);
 const i=D.contexts.indexOf(+c);$('storage-slider').value=i;$('storage-slider').style.setProperty('--storage-progress',100*i/(D.contexts.length-1)+'%');$('storage-slider').setAttribute('aria-valuetext',num(+c,0)+' input tokens, '+context(c)+' measured context');
 if(!d){$('storage-total-value').textContent='Unsupported';$('storage-total-unit').textContent='';$('storage-percentage-bar').replaceChildren();$('storage-percentage-labels').replaceChildren();$('storage-percentage-bar').setAttribute('aria-label','No 128K measurement for Llama');$('storage-explanation').textContent='This Llama deployment cannot fit a 128K prompt plus generation within its supported context limit. Move the slider left to a measured point.';$('storage-measurement-note').textContent='No extrapolated or zero-filled memory value is shown.';return}
 const parts=d.storage_parts,keys=storageOrder.filter(k=>parts[k]>0);
 $('storage-total-value').textContent=perGpu?num(d.median*1024/8,1):num(d.median,3);$('storage-total-unit').textContent=perGpu?'MiB per GPU':'GiB across 8 GPUs';
 $('storage-percentage-bar').innerHTML=keys.map(k=>'<span style="flex-basis:'+100*parts[k]/d.median+'%;background:'+storagePalette[k]+'" title="'+esc(storageLabel(m,k))+': '+esc(storagePercent(parts[k],d.median))+' · '+storageAmount(parts[k],perGpu)+'"></span>').join('');
 $('storage-percentage-bar').setAttribute('aria-label',m.name+' attention-state storage: '+keys.map(k=>storageLabel(m,k)+' '+storagePercent(parts[k],d.median)).join(', '));
 if(!keys.includes(storageCategory))storageCategory=keys[0];
 $('storage-percentage-labels').innerHTML=keys.map(k=>'<button type="button" data-storage-category="'+k+'" aria-pressed="'+(k===storageCategory)+'"><i class="swatch" style="background:'+storagePalette[k]+'"></i><code>'+esc(storageLabel(m,k))+'</code><strong>'+esc(storagePercent(parts[k],d.median))+'</strong></button>').join('');
 $('storage-explanation').innerHTML='<strong>'+esc(storageLabel(m,storageCategory))+' · '+storageAmount(parts[storageCategory],perGpu)+'.</strong> '+esc(storageDescription[storageCategory]);
 $('storage-measurement-note').textContent='Measured '+context(c)+' prompt; snapshot at '+num(d.live_tokens,0)+' live tokens. Three-repeat median. '+(keys.includes('sconv')?'The short-conv, matrix and padding split is calculated from recorded runtime shapes and live pages. ':'')+'Select a component for its meaning and size.';
}
function exportStorage(){
 const {m,c,d,perGpu}=storageSelection();
 let svg='<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="410" viewBox="0 0 1080 410"><title>'+esc(m.name)+' attention storage at '+context(c)+'</title><rect width="100%" height="100%" fill="#fbfcf8"/><g font-family="Segoe UI, Arial, sans-serif" fill="#172a2a"><text x="35" y="45" font-size="23">'+esc(m.name)+' · one decoding request</text><text x="35" y="82" font-size="18">Context: '+num(+c,0)+' input tokens</text><text x="35" y="123" font-size="27">'+(d?storageAmount(d.median,perGpu)+(perGpu?' per GPU':' across 8 GPUs'):'Unsupported at this context')+'</text>';
 if(d){let x=35;const keys=storageOrder.filter(k=>d.storage_parts[k]>0);keys.forEach(k=>{const w=1010*d.storage_parts[k]/d.median;svg+='<rect x="'+x+'" y="165" width="'+w+'" height="35" fill="'+storagePalette[k]+'"/>';x+=w});keys.forEach((k,i)=>{const x=35+(i%3)*340,y=235+Math.floor(i/3)*35;svg+='<rect x="'+x+'" y="'+(y-12)+'" width="12" height="12" fill="'+storagePalette[k]+'"/><text x="'+(x+20)+'" y="'+y+'" font-size="15">'+esc(storageLabel(m,k))+' '+esc(storagePercent(d.storage_parts[k],d.median))+'</text>'})}
 return svg+'<text x="35" y="350" font-size="12">H100 settled-decode occupied state · three-repeat median · per-GPU value is rank average</text><text x="35" y="375" font-size="12">Source: '+D.study+'; component splits derived from live pages and runtime specs.</text></g></svg>';
}

function renderMemory(){
 renderStorage();
 const c=$('memory-context').value,n=+$('memory-requests').value,rs=M.map(m=>({m,d:m.memory[c]})),max=Math.max(...rs.filter(r=>r.d).map(r=>r.d.median*n));
 $('memory-bars').innerHTML=rs.map(({m,d})=>'<div class="bar-row"><span>'+esc(m.name)+'</span><div class="bar-track" role="img" aria-label="'+esc(m.name)+' '+(d?num(d.median*n)+' GiB':'unsupported')+'">'+(d?categories.map((cat,i)=>'<span class="bar-fill" style="background:'+colors[i]+';width:'+100*d[cat]*n/max+'%" title="'+labels[i]+': '+num(d[cat]*n,3)+' GiB"></span>').join(''):'')+'</div><span class="bar-value">'+(d?num(d.median*n)+' GiB':'unsupported')+'</span></div>').join('');
 $('memory-table').innerHTML=table(['Deployment',...labels.map(x=>x+' GiB'),'Total GiB','Actual live tokens'],rs.map(({m,d})=>d?[esc(m.name),...categories.map(k=>num(d[k]*n,3)),num(d.median*n,3),num(d.live_tokens,0)]:[esc(m.name),'<span>unsupported at 128K</span>',...Array(6).fill('—')]));
 $('memory-context-note').textContent=(n===1?'Measured one-request state. ':'Calculated eight-request scaling; no admission or concurrency claim. ')+ 'Prompt: '+context(c)+'. Actual live lengths reflect generated tokens at the settled snapshot. All values sum eight GPUs.';
}
function renderLayers(){
 const m=M.find(m=>m.id===$('layer-model').value),c=$('layer-context').value;
 layerIndex=Math.min(layerIndex,m.layers-1);
 $('layer-tiles').innerHTML=m.layer_data.map(l=>'<button type="button" data-layer="'+l.index+'" aria-pressed="'+(l.index===layerIndex)+'" style="background:'+colorType(l.type)+'" title="Layer '+l.index+': '+esc(l.type)+'">'+l.index+'</button>').join('');
 if(!m.memory[c]){$('layer-detail').textContent='Unsupported: no 128K live-state measurement for Llama in this deployment.';$('layer-subtotals').replaceChildren();return}
 const l=m.layer_data[layerIndex],d=l.memory[c];
 $('layer-detail').innerHTML='<strong>Layer '+l.index+' · '+esc(l.type)+'</strong><p>Owns '+num(d.mib,3)+' MiB across the node at '+context(c)+'.</p>'+categories.filter(k=>d.categories[k]>0).map(k=>esc(labels[categories.indexOf(k)])+': '+num(d.categories[k],3)+' MiB').join(' · ')+(l.references.length?'<p class="data-notice">Shared references: '+l.references.map(esc).join('; ')+'. Shared bytes are counted at the owner.</p>':'');
 const bytype={};m.layer_data.forEach(l=>{const t=bytype[l.type]||(bytype[l.type]={n:0,mib:0});t.n++;t.mib+=l.memory[c].mib});
 $('layer-subtotals').innerHTML=table(['Layer type','Count','Owned MiB, all GPUs','Average MiB/layer'],Object.entries(bytype).map(([k,v])=>[esc(k),v.n,num(v.mib),num(v.mib/v.n)]).concat([['Total',m.layers,num(m.memory[c].median*1024), 'Shared consumers add no duplicate region']]));
}
function lineChart(){
 const b=$('decode-batch').value,W=850,H=365,L=65,R=25,T=25,B=48,max=Math.ceil(Math.max(...M.flatMap(m=>D.contexts.map(c=>m.decode[c+':'+b]?.max||0)))/2)*2;
 const X=c=>L+Math.log2(c/1024)/7*(W-L-R),Y=v=>H-B-v/max*(H-T-B);
 let out='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 '+W+' '+H+'" role="img" aria-labelledby="decode-title decode-desc"><title id="decode-title">Measured decode-attention module time, batch '+b+'</title><desc id="decode-desc">Six model series across 1K, 16K, 64K and 128K contexts. Medians with min/max whiskers. Llama has no 128K measurement. Exact values are in the adjacent table.</desc><rect width="100%" height="100%" fill="#fff"/><g font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#576969">';
 for(let i=0;i<=4;i++){let v=max*i/4,y=Y(v);out+='<line x1="'+L+'" y1="'+y+'" x2="'+(W-R)+'" y2="'+y+'" stroke="#e2e7df"/><text x="'+(L-12)+'" y="'+(y+4)+'" text-anchor="end">'+num(v,1)+'</text>'}
 out+='<text x="'+L+'" y="13">ms / decode step</text>';
 for(const c of D.contexts)out+='<text x="'+X(c)+'" y="'+(H-16)+'" text-anchor="middle">'+context(c)+'</text>';
 for(const m of M){const pts=D.contexts.filter(c=>m.decode[c+':'+b]).map(c=>[c,m.decode[c+':'+b]]);
 out+='<polyline points="'+pts.map(([c,d])=>X(c)+','+Y(d.median)).join(' ')+'" fill="none" stroke="'+m.color+'" stroke-width="2.7"/>';
 for(const [c,d]of pts)out+='<path d="M'+X(c)+' '+Y(d.min)+'V'+Y(d.max)+'" stroke="'+m.color+'" stroke-width="2"/><circle cx="'+X(c)+'" cy="'+Y(d.median)+'" r="4.8" fill="'+m.color+'" stroke="white" stroke-width="1.4" tabindex="0"><title>'+esc(m.name)+' '+context(c)+': '+num(d.median,3)+' ms; range '+num(d.min,3)+'–'+num(d.max,3)+' ms</title></circle>';
 }
 $('decode-line').innerHTML=out+'</g></svg>';
 $('decode-table').innerHTML=table(['Deployment',...D.contexts.map(c=>context(c)+' ms [min–max]')],M.map(m=>[esc(m.name),...D.contexts.map(c=>{const d=m.decode[c+':'+b];return d?num(d.median,3)+' ['+num(d.min,3)+'–'+num(d.max,3)+']':'unsupported'})]));
}
function overhead(){
 const rs=M.map(m=>({name:m.name,value:m.decode['65536:8'].median,min:m.decode['65536:8'].sensitivity,max:m.decode['65536:8'].median,color:m.color}));barRows('overhead-bars',rs,Math.max(...rs.map(r=>r.value)),'ms');
 $('overhead-table').innerHTML=table(['Deployment','Measured attention ms','Control sensitivity ms','Probed model ms','Attention share'],M.map(m=>{const d=m.decode['65536:8'];return [esc(m.name),num(d.median),num(d.sensitivity)+'–'+num(d.median),num(d.model),num(100*d.median/d.model,1)+'%']}));
}
function renderPrefill(){
 const c=$('prefill-context').value;
 const rs=M.map(m=>{const d=m.prefill[c];return {name:m.name,color:m.color,value:d?.median??null,min:d?.min,max:d?.max}});
 barRows('prefill-attention',rs,Math.max(...rs.map(r=>r.max||0)),'ms',1);
 const models=M.map(m=>({name:m.name,color:m.color,value:m.prefill[c]?.model??null}));
 barRows('prefill-model',models,Math.max(...models.map(r=>r.value||0)),'ms',1);
 $('prefill-table').innerHTML=table(['Deployment','Attention ms [min–max]','Model ms','Executed positions × layer count'],M.map(m=>{const d=m.prefill[c];return [esc(m.name),d?num(d.median,1)+' ['+num(d.min,1)+'–'+num(d.max,1)+']':'unsupported',d?num(d.model,1):'—',d?Object.entries(d.positions).map(([p,n])=>num(+p,0)+' × '+n).join('; '):'—']}));
}
function deployment(){
 $('deployment-table').innerHTML=table(['Checkpoint / immutable revision','Weight inventory','KV choice','Actual state dtype evidence'],M.map(m=>['<a href="https://huggingface.co/'+m.repo+'/tree/'+m.revision+'" target="_blank" rel="noopener">'+esc(m.repo)+'</a><br><code>'+m.revision+'</code>',esc(m.weight_dtype),esc(m.kv),esc(m.dtype_evidence)]));
 $('first-chunk-table').innerHTML=table(['Deployment','Attention ms [min–max]','Model ms'],M.map(m=>{const d=m.prefill['1024'];return [esc(m.name),num(d.median,1)+' ['+num(d.min,1)+'–'+num(d.max,1)+']',num(d.model,1)]}));
}
function exportBars(kind){
 const c=kind==='memory'?$('memory-context').value:$('prefill-context').value,n=kind==='memory'?+$('memory-requests').value:1;
 const values=M.map(m=>({m,d:kind==='memory'?m.memory[c]:m.prefill[c]})),max=Math.max(...values.map(r=>(r.d?.median||0)*n));
 let out='<svg xmlns="http://www.w3.org/2000/svg" width="960" height="450" viewBox="0 0 960 450"><title>'+esc(kind)+' at '+context(c)+'</title><rect width="100%" height="100%" fill="#fff"/><g font-family="Segoe UI, Arial, sans-serif" font-size="15">';
 out+='<text x="25" y="35">H100 · '+(kind==='memory'?'occupied attention state, GiB':'prefill attention-module time, ms')+' · '+context(c)+(n>1?' · calculated '+n+' requests':' · three-repeat medians')+'</text>';
 values.forEach(({m,d},i)=>{let y=75+i*48;out+='<text x="25" y="'+(y+18)+'">'+esc(m.name)+'</text>';if(d){let x=210;for(const [cat,color]of(kind==='memory'?categories.map((k,i)=>[k,colors[i]]):[['median',m.color]])){let w=d[cat]*n/max*580;out+='<rect x="'+x+'" y="'+y+'" width="'+w+'" height="25" fill="'+color+'"/>';x+=w}out+='<text x="810" y="'+(y+18)+'">'+num(d.median*n)+'</text>'}else out+='<text x="210" y="'+(y+18)+'">unsupported</text>'});
 if(kind==='memory'){let lx=25;categories.forEach((cat,i)=>{out+='<rect x="'+lx+'" y="373" width="10" height="10" fill="'+colors[i]+'"/><text x="'+(lx+16)+'" y="383" font-size="12">'+labels[i]+'</text>';lx+=[125,120,215,130,140][i]})}
 return out+'<text x="25" y="410" font-size="12">Source: '+D.study+' · '+(kind==='memory'?'All 8 GPUs; settled decode. Excludes weights/reserved pool.':'CUDA-event module sums. Includes probe overhead; not critical-path latency.')+'</text></g></svg>';
}
function saveChart(kind){
 let svg;if(kind==='decode'){const chart=document.querySelector('#decode-line svg').cloneNode(true);chart.setAttribute('viewBox','0 0 850 450');chart.setAttribute('width','960');chart.setAttribute('height','508');const group=document.createElementNS('http://www.w3.org/2000/svg','g');group.setAttribute('font-family','Segoe UI, Arial, sans-serif');group.setAttribute('font-size','11');group.innerHTML=M.map((m,i)=>'<rect x="'+(20+(i%3)*275)+'" y="'+(375+Math.floor(i/3)*23)+'" width="10" height="10" fill="'+m.color+'"/><text x="'+(36+(i%3)*275)+'" y="'+(384+Math.floor(i/3)*23)+'">'+esc(m.name)+'</text>').join('')+'<text x="20" y="435">Source: '+D.study+' · module sums, 3-repeat min/median/max; includes probe overhead.</text>';chart.appendChild(group);svg=new XMLSerializer().serializeToString(chart)}else svg=kind==='memory'?exportStorage():exportBars(kind);const blob=new Blob([svg],{type:'image/svg+xml'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='h100-'+kind+'.svg';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
$('model-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-model]');if(b){selected=b.dataset.model;renderArchitecture()}});
$('architecture-sequence').addEventListener('click',e=>{const tile=e.target.closest('[data-architecture-layer]');if(!tile)return;layerIndex=+tile.dataset.architectureLayer;$('memory').scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});$('layer-model').value=selected;renderLayers()});
$('layer-model').innerHTML=M.map(m=>'<option value="'+m.id+'">'+esc(m.name)+'</option>').join('');
$('layer-tiles').addEventListener('click',e=>{const b=e.target.closest('[data-layer]');if(b){layerIndex=+b.dataset.layer;renderLayers()}});
$('storage-model').innerHTML=M.map(m=>'<option value="'+m.id+'"'+(m.id==='mimo-v26'?' selected':'')+'>'+esc(m.name)+'</option>').join('');
$('storage-slider').addEventListener('input',()=>{$('memory-context').value=D.contexts[+$('storage-slider').value];renderMemory()});
for(const id of ['storage-model','storage-scope'])$(id).addEventListener('change',renderStorage);
$('storage-percentage-labels').addEventListener('click',e=>{const b=e.target.closest('[data-storage-category]');if(b){storageCategory=b.dataset.storageCategory;renderStorage()}});
for(const id of ['memory-context','memory-requests'])$(id).addEventListener('change',renderMemory);
for(const id of ['layer-model','layer-context'])$(id).addEventListener('change',renderLayers);
$('decode-batch').addEventListener('change',lineChart);$('prefill-context').addEventListener('change',renderPrefill);
document.querySelectorAll('[data-export]').forEach(b=>b.addEventListener('click',()=>saveChart(b.dataset.export)));
legend('memory-legend',labels.map((l,i)=>[l,colors[i]]));legend('decode-legend',M.map(m=>[m.name,m.color]));
renderArchitecture();renderParameters();renderMemory();renderLayers();lineChart();overhead();renderPrefill();deployment();
})();
