"""Accessible vector schematics: shapes explain mechanisms, not measured allocations."""
from html import escape
INK='#172a2a';GREEN='#3e7057';BLUE='#5076a6';ORANGE='#b7783d';PURPLE='#7b6aa6';TEAL='#228b8b';ROSE='#ba6b85'
def text(x,y,s,size=16,color=INK,weight=400,anchor='start'):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(s)}</text>'
def box(x,y,w,h,label,color=GREEN,sub=None):
    a=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{color}" fill-opacity=".09" stroke="{color}" stroke-width="1.5"/>'
    a+=text(x+w/2,y+h/2+(0 if sub else 6),label,17,color,600,'middle')
    if sub:a+=text(x+w/2,y+h/2+23,sub,12,color,400,'middle')
    return a
def arrow(x1,y1,x2,y2,color=GREEN):
    return f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#arrow)"/>'
def wrap(title,desc,body,w=960,h=450):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 1 L7 4 L0 7" fill="none" stroke="#576969" stroke-width="1.5"/></marker></defs><rect width="100%" height="100%" rx="12" fill="#fbfcf8"/><g font-family="Segoe UI, Arial, sans-serif">{body}</g></svg>'
def matrix(kind,x=45,y=100):
    out=text(x,y-18,'Which past tokens can a query read?',16,INK,600)
    for q in range(9):
        for k in range(9):
            on=k<=q and (kind=='full' or (kind=='window' and q-k<3))
            color=({'full':BLUE,'window':ORANGE}[kind] if on else '#e7ece6')
            out+=f'<rect x="{x+k*23}" y="{y+q*23}" width="19" height="19" rx="3" fill="{color}"/>'
    return out+text(x,y+232,'Earlier token →',13)+text(x+222,y+100,'query',12)
def history(x,y,n=12,color=BLUE):
    return ''.join(f'<rect x="{x+i*24}" y="{y}" width="18" height="27" rx="3" fill="{color}" fill-opacity="{.32+i*.045}"/>' for i in range(n))
def diagrams():
    d={}
    b=text(40,42,'LLAMA 3.1 / KEEP THE HISTORY',22,GREEN,600)+matrix('full')
    b+=box(400,103,210,70,'64 query heads',BLUE,'queries for the new token')+arrow(610,138,663,138)
    b+=box(670,103,230,70,'8 KV heads',BLUE,'shared across query heads')
    b+=history(480,231,12)+text(480,216,'A growing K/V history at each layer',15)
    b+=box(438,304,410,65,'Repeat at all 80 layers',GREEN,'GQA shares heads; history still grows')
    b+=text(40,413,'Read broadly · store independently · state grows with context × layers',16)
    d['llama-31-70b']=wrap('Llama 3.1: full-history grouped-query attention','A causal attention matrix and 64 query heads sharing eight KV heads. Every one of 80 layers keeps growing history.',b)
    b=text(40,42,'MIMO / LOOK NEARBY, THEN LOOK FAR',22,ORANGE,600)+matrix('window')
    b+=box(390,96,225,88,'39 local layers',ORANGE,'128-token window')+box(670,96,240,88,'9 global layers',BLUE,'full-history GQA')
    b+=arrow(500,184,500,226)+arrow(790,184,790,226)
    b+=history(452,242,4,ORANGE)+history(644,242,10,BLUE)
    b+=text(420,299,'Old window entries expire',14,ORANGE)+text(670,299,'History keeps growing',14,BLUE)
    b+=box(415,327,455,49,'Alternate local work with global access',ORANGE)
    b+=text(40,413,'Global layers keep long-range access; local state stays bounded.',16)
    d['mimo-v26']=wrap('MiMo: sliding windows with periodic global attention','Thirty-nine layers use a 128-token window and nine retain the full history. The matrix illustrates the local mechanism, not all MiMo layers.',b)
    b=text(40,42,'DEEPSEEK V4 / COMPRESS BEFORE LOOKING BACK',22,BLUE,600)
    b+=text(48,98,'Past tokens',16,INK,600)+history(48,115,12)
    b+=arrow(186,153,186,192)+box(49,202,275,66,'Compress the history',BLUE,'4× or 128×, depending on layer')
    b+=arrow(324,234,382,234)+box(393,200,238,70,'Compressed KV',BLUE,'fewer long-range entries')
    b+=box(395,97,234,61,'Index the 4× path',PURPLE)+arrow(512,158,512,200,PURPLE)
    b+=arrow(631,235,679,235)+box(690,199,218,73,'Attention',GREEN,'compressed history')
    b+=box(48,308,275,63,'Recent 128-token window',ORANGE)+arrow(323,340,801,290,ORANGE)
    b+=text(40,413,'Smaller history still needs compression, selection and temporary state.',16)
    d['v4-0731']=wrap('DeepSeek V4: compressed history and sparse retrieval','Two window-only layers, 21 layers with fourfold compression and an indexer, and 20 with 128-fold compression. Recent-window access is retained.',b)
    for mid,title,recurrent,sparse,name in [('qwen-38-fp8','QWEN3.8',36,12,'Gated DeltaNet'),('glm-53','GLM-5.3',34,11,'Kimi Delta Attention')]:
        b=text(40,42,f'{title} / UPDATE A STATE, KEEP A RETRIEVAL PATH',21,TEAL,600)+text(45,95,'Most layers',16,INK,600)
        for i in range(4):
            b+=box(45+i*133,112,101,52,f'token {i+1}',TEAL)
            if i<3:b+=arrow(146+i*133,138,174+i*133,138,TEAL)
        b+=box(113,216,342,76,'Fixed-size recurrent matrix',TEAL,f'{recurrent} {name} layers')+arrow(360,166,325,216,TEAL)
        b+=box(610,111,300,80,f'{sparse} sparse-attention layers',PURPLE,'a separate growing history')
        b+=text(615,225,'Indexer → selected past entries',16,PURPLE)+history(614,246,12,PURPLE)
        b+=box(615,318,296,49,'KV / latents + index + tail',PURPLE)
        b+=text(40,413,'The recurrent matrix is bounded. The sparse layers still store history.',16)
        d[mid]=wrap(f'{title}: recurrence with sparse history access',f'{recurrent} recurrent layers update a fixed-size matrix; {sparse} sparse layers retain growing state and selection metadata. The two paths occur in different layers.',b)
    b=text(40,42,'DEEPSEEK V4.1 / LET LAYERS SHARE HISTORY',22,ROSE,600)
    b+=box(52,110,268,78,'Four KV source layers',ROSE,'zero-based layers 2, 8, 14, 20')
    b+=arrow(320,149,386,149,ROSE)+box(398,99,223,100,'Shared KV buffers',ROSE,'count each owner once')
    for i in range(3):
        y=89+i*98;b+=box(713,y,202,62,f'Consumer layer {i+1}',BLUE)+arrow(621,149,709,y+31,ROSE)
    b+=box(51,281,280,72,'Local windows remain',ORANGE,'128 recent tokens at each layer')
    b+=box(398,300,240,58,'Indexer sources also share',PURPLE)
    b+=text(40,413,'Consumer arrows summarize config references; bytes come from live-state records.',15)
    d['v41']=wrap('DeepSeek V4.1: shared compressed KV across layers','Four layers own long-range KV buffers. Many later layers read those buffers while retaining local windows. The consumer mapping is inferred from the checkpoint config, not directly traced.',b)
    b=text(40,45,'THE ATTENTION JOURNEY',24,GREEN,600)+text(40,77,'Six deployments. Several ways to carry the past forward.',17)
    cards=[(40,111,'Llama','Keep every token',BLUE),(350,111,'MiMo','Keep nearby + global',ORANGE),(660,111,'DeepSeek V4','Compress + select',PURPLE),(40,280,'Qwen','Recur + retrieve',TEAL),(350,280,'GLM','Recur + sparse latents',TEAL),(660,280,'DeepSeek V4.1','Share between layers',ROSE)]
    for x,y,label,sub,color in cards:b+=box(x,y,245,99,label,color,sub)
    d['journey']=wrap('An attention journey from full history to hybrid, compressed, recurrent and shared state','Six labeled cards introduce full-history, local/global, compressed, recurrent and shared attention mechanisms.',b)
    return d
