"""Step 2: Enrich Gold — LLM审核FP并补充漏标。"""
import json
from pathlib import Path

from src.model_platform.llm import create_llm_client
from src.skill_extraction.soft_skill_llm_extractor import extract_soft_skills


def norm(s):
    return s.strip().casefold()


def main():
    with open("output/skill_extraction/eval/soft_skill_gold_normalized.jsonl", "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    client = create_llm_client(backend="external_api")
    enriched = 0

    for i, s in enumerate(samples):
        pred = extract_soft_skills(text=s["text"], llm_client=client)
        gold_names = {norm(g["name"]) for g in s["gold_soft_skills"]}

        extra = [
            p for p in pred
            if not any(
                norm(p["name"]) == gn or norm(p["name"]) in gn or gn in norm(p["name"])
                for gn in gold_names
            )
        ]
        if not extra:
            continue

        extra_list = "\n".join(f"- {e['name']} [{e['dimension']}]" for e in extra)
        try:
            resp = client.complete_json(
                system_prompt="你是HR专家。判断以下软技能是否在岗位描述中有明确依据。只输出JSON数组。",
                user_prompt=f"岗位:\n{s['text'][:1500]}\n\n候选:\n{extra_list}\n\n有依据的技能:",
                strength="cheap",
                max_output_tokens=500,
            )
            if isinstance(resp, list):
                confirmed = {norm(n) for n in resp if isinstance(n, str)}
                for e in extra:
                    if norm(e["name"]) in confirmed:
                        s["gold_soft_skills"].append(
                            {"name": e["name"], "dimension": e["dimension"]}
                        )
                        enriched += 1
        except Exception:
            pass

        if i % 50 == 0:
            print(f"  {i}/{len(samples)}")

    total = sum(len(s["gold_soft_skills"]) for s in samples)
    print(f"Enriched: +{enriched}, total labels: {total}")

    out = "output/skill_extraction/eval/soft_skill_gold_enriched.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
