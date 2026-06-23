"""Qwen3-8B 50并发 5000条 测速"""
import json, time, torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from src.db.postgres import create_pg_engine
from sqlalchemy import text

m = SentenceTransformer('output/occupation_retrieval/rag_round2_training/bge-large-round2-finetuned-v5', device='cuda')
e = create_pg_engine()
with e.connect() as c:
    r = c.execute(text('select code, title, "desc", tasks from public.occ_dict_unified where node_type = :t'), {'t': 'occupation_leaf'})
    occ_texts = []; occ_codes = []
    for row in r.mappings():
        code = str(row['code']).strip(); title = str(row['title']).strip()
        desc = str(row.get('desc','')).strip(); tasks = str(row.get('tasks','')).strip()
        parts = [title]
        if desc and desc != 'nan': parts.append(f'定义：{desc}')
        if tasks and tasks != 'nan': parts.append(f'任务：{tasks}')
        occ_texts.append('。'.join(parts)); occ_codes.append(code)
with torch.no_grad():
    occ_emb = m.encode(occ_texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True, convert_to_tensor=True)

done = set()
try:
    with e.connect() as c:
        rows = c.execute(text('select recruitment_record_id from public.deepseek_full_label')).fetchall()
        done = {str(r[0]) for r in rows}
except:
    pass

tasks = []
with e.connect() as c:
    rows = c.execute(text("select recruitment_record_id, job_title, coalesce(requirements_text,'') as req, coalesce(rag_query_text,'') as rag from public.job_description_parsed where requirements_text is not null and requirements_text != '' order by recruitment_record_id limit 50000")).mappings()
    for row in rows:
        rid = str(row['recruitment_record_id'])
        if rid in done: continue
        tasks.append({'rid': rid, 'jt': str(row['job_title'] or ''), 'req': str(row['req']), 'rag': str(row['rag'])})
        if len(tasks) >= 5000: break
print(f'跳过{len(done)}条, 新取{len(tasks)}条')

queries = [t['rag'] if t['rag'] else f"{t['jt']} {t['req']}"[:500] for t in tasks]
top5_all = []
for s in range(0, len(queries), 500):
    e2 = min(s+500, len(queries))
    with torch.no_grad():
        qe = m.encode(queries[s:e2], batch_size=64, normalize_embeddings=True, show_progress_bar=True, convert_to_tensor=True)
    sim = torch.mm(qe, occ_emb.T)
    _, idxs = torch.topk(sim, k=5, dim=1)
    for idx in idxs.cpu().tolist():
        top5_all.append([(occ_codes[i], occ_texts[i][:occ_texts[i].index('。')] if '。' in occ_texts[i] else occ_texts[i][:40]) for i in idx])

qwen = OpenAI(api_key='x', base_url='http://127.0.0.1:8101/v1')
S = '你是职业分类专家。从5个候选职业中选最匹配的一个。如果都不合适选NONE。只输出严格JSON。'

def call_one(task, top5):
    u = f'岗位：{task["jt"]}\n描述：{task["req"][:2000]}\n候选：A[{top5[0][0]}]{top5[0][1]} B[{top5[1][0]}]{top5[1][1]} C[{top5[2][0]}]{top5[2][1]} D[{top5[3][0]}]{top5[3][1]} E[{top5[4][0]}]{top5[4][1]}\n输出JSON：{{"best_candidate":"A"|"B"|"C"|"D"|"E"|"NONE","confidence":0.0}}'
    try:
        r = qwen.chat.completions.create(
            model='Qwen3-8B', messages=[{'role':'system','content':S},{'role':'user','content':u}],
            response_format={'type':'json_object'}, temperature=0.0, max_tokens=128, timeout=120,
            extra_body={'enable_thinking': False})
        return json.loads(r.choices[0].message.content or '{}')
    except:
        return None

ok = 0; fail = 0; none = 0; confs = []
t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=50) as ex:
    futs = {ex.submit(call_one, t, top5): i for i, (t, top5) in enumerate(zip(tasks, top5_all))}
    for f in as_completed(futs):
        p = f.result()
        if p and p.get('best_candidate') in 'ABCDENONE':
            ok += 1
            c = float(p.get('confidence', 0)); confs.append(c)
            if p['best_candidate'] == 'NONE': none += 1
        else: fail += 1
        if (ok+fail) % 500 == 0: print(f'  {ok+fail}/{len(tasks)}')

el = time.perf_counter() - t0
print(f'\nQwen3-8B 50并发 5000条: {el:.0f}s, {5000/el:.1f}/s')
print(f'成功:{ok} 失败:{fail} NONE率:{none/ok*100:.1f}%')
print(f'置信度: >=0.9:{sum(1 for c in confs if c>=0.9)} 0.7-0.9:{sum(1 for c in confs if 0.7<=c<0.9)} <0.7:{sum(1 for c in confs if c<0.7)}')
