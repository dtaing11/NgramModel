#!/usr/bin/env python3
from openai import OpenAI
from dotenv import load_dotenv
import os, json, time, urllib.request, urllib.error, datetime, textwrap
import chromadb
from chromadb.config import Settings

SquaAd = "dev-v2.0.json"
FIVEHUNDREDQUESTION = "500-Question.jsonl"
GPT5_Nano = "gpt-5-nano"
GPT5_Mini = "gpt-5-mini"
QWEN_MODEL = "qwen/qwen3-8b"

BATCH_IN = "gpt5nano_batch_input.jsonl"
NANO_BATCH_OUT_RAW = "gpt5nano_batch_output.jsonl"
NANO_PREDS = "gpt5nano_answers.jsonl"
QWEN_PREDS = "qwen3_8b_answers.jsonl"

TODAY = datetime.date.today().isoformat()
NANO_JUDGE_IN = f"judge_gpt5nano_{TODAY}.jsonl"
QWEN_JUDGE_IN = f"judge_qwen_{TODAY}.jsonl"
NANO_JUDGE_OUT = f"gpt-5-nano-RAG-{TODAY}-hw4.json"
QWEN_JUDGE_OUT = f"qwen3-8b-RAG-{TODAY}-hw4.json"
ENDPOINT = "/v1/responses"

SYSTEM_PROMPT_RAG = (
    "You are a careful, concise QA assistant. Use only the provided sources. "
    "If the answer cannot be found, reply exactly: \"I don't know\"."
)

def make_user_prompt(question: str, sources: list[str]) -> str:
    src_lines = []
    for i, chunk in enumerate(sources, 1):
        chunk = " ".join(chunk.split())
        src_lines.append(f"[{i}] {chunk}")
    src_block = "\n".join(src_lines)
    return textwrap.dedent(f"""\
        Use ONLY the following sources to answer. If uncertain, say "I don't know".

        Question:
        {question}

        Sources:
        {src_block}

        Rules:
        - Cite by bracket number when helpful, e.g., [2].
        - Prefer precise, minimal answers.
        - If sources conflict or are irrelevant, say "I don't know".
        - Do not invent facts beyond the sources.
    """)

JUDGE_PROMPT_TMPL = (
    "You are a teacher tasked with determining whether a student’s answer to a question was correct,\n"
    "based on a set of possible correct answers. You must only use the provided possible correct answers\n"
    "to determine if the student’s response was correct.\n"
    "Question: {question}\n"
    "Student’s Response: {student_response}\n"
    "Possible Correct Answers:\n"
    "{correct_answers}\n"
    "Your response should only be a valid Json as shown below:\n"
    "{\n"
    "\"explanation\": \"A short explanation of why the student’s answer was correct or incorrect.\",\n"
    "\"score\": true or false\n"
    "}\n"
    "Your response: "
)

judge_schema = {
    "type": "object",
    "properties": {"explanation": {"type": "string"}, "score": {"type": "boolean"}},
    "required": ["explanation", "score"],
    "additionalProperties": False
}

# ========= RAG: ChromaDB (MINIMAL helper additions) =========
import chromadb
from chromadb.config import Settings

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "squad_chunks")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

def _get_chroma_client():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))

def _get_collection(client):
    return client.get_or_create_collection(CHROMA_COLLECTION)

def _oa_embed(texts: list[str]) -> list[list[float]]:
    oa = OpenAI()
    out = []
    B = 512
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        res = oa.embeddings.create(model=EMBED_MODEL, input=batch).data
        out.extend([e.embedding for e in res])
    return out

def seed_chroma_if_needed(col):
    if col.count() > 0:
        return
    if not os.path.exists(SquaAd):
        print("[chroma] dev-v2.0.json not found; skipping auto-seed")
        return
    with open(SquaAd, "r", encoding="utf-8") as f:
        dev = json.load(f)
    docs, ids = [], []
    idx = 0
    for art in dev["data"]:
        for para in art["paragraphs"]:
            ctx = " ".join(para.get("context","").split())
            size, overlap = 420, 60
            for start in range(0, len(ctx), size - overlap):
                chunk = ctx[start:start+size].strip()
                if len(chunk) < 100:
                    continue
                docs.append(chunk); ids.append(f"p{idx}"); idx += 1
    if not docs:
        return
    embeds = _oa_embed(docs)
    col.add(documents=docs, embeddings=embeds, ids=ids)
    print(f"[chroma] collection ready: {col.name} (count={col.count()})")

def _embed_query(q: str) -> list[float]:
    oa = OpenAI()
    e = oa.embeddings.create(model=EMBED_MODEL, input=q)
    return e.data[0].embedding

def get_sources(col, question: str, k: int = 5) -> list[str]:
    qvec = _embed_query(question)
    res = col.query(query_embeddings=[qvec], n_results=k, include=["documents"])
    docs = (res.get("documents") or [[]])[0]
    return docs

def extractQuestion(dev):
    count = 0
    with open(FIVEHUNDREDQUESTION, "w", encoding="utf-8") as w:
        for art in dev["data"]:
            for para in art["paragraphs"]:
                for qa in para["qas"]:
                    if qa.get("is_impossible", False):
                        continue
                    answers = qa.get("answers", [])
                    if not answers:
                        continue
                    gold = answers[0]["text"]
                    item = {"id": qa["id"], "question": qa["question"].strip(), "gold": gold.strip()}
                    w.write(json.dumps(item) + "\n")
                    count += 1
                    if count >= 500:
                        return

def record (question_id, question):
    return {
        "custom_id": f"g5nano::{question_id}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": GPT5_Nano,
            "reasoning": {"effort": "minimal"},
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT_RAG},
                {"role": "user", "content": f"Question: {question}"},
            ],
        }
    }

if __name__ == "__main__":
    load_dotenv(".env")

    if not os.path.exists(FIVEHUNDREDQUESTION):
        if not os.path.exists(SquaAd):
            raise FileNotFoundError(f"Put SQuAD dev set at: {SquaAd}")
        with open(SquaAd, "r", encoding="utf-8") as f:
            dev = json.load(f)
        extractQuestion(dev)
        print(f"[extract] wrote -> {FIVEHUNDREDQUESTION}")

    chroma_client = _get_chroma_client()
    col = _get_collection(chroma_client)
    seed_chroma_if_needed(col)

    with open(FIVEHUNDREDQUESTION, "r", encoding="utf-8") as f, open(BATCH_IN, "w", encoding="utf-8") as w:
        for line in f:
            ex = json.loads(line)
            chunks = get_sources(col, ex["question"], k=5)
            rag_user = make_user_prompt(ex["question"], chunks)
            job = {
                "custom_id": f"g5nano::{ex['id']}",
                "method": "POST",
                "url": ENDPOINT,
                "body": {
                    "model": GPT5_Nano,
                    "reasoning": {"effort": "minimal"},
                    "input": [
                        {"role": "system", "content": SYSTEM_PROMPT_RAG},
                        {"role": "user", "content": rag_user},
                    ],
                }
            }
            w.write(json.dumps(job) + "\n")
    print(f"[nano] batch input (RAG) -> {BATCH_IN}")

    client = OpenAI()
    with open(BATCH_IN, "rb") as fh:
        up = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(input_file_id=up.id, endpoint="/v1/responses", completion_window="24h")
    print(f"[batch] created id={batch.id} status={batch.status}")

    printed_status = set()
    while True:
        time.sleep(5)
        b = client.batches.retrieve(batch.id)
        if b.status in ("completed", "failed", "expired", "cancelled"):
            print(f"[batch] final status = {b.status}")
            if b.status != "completed":
                raise RuntimeError(f"Batch ended with status = {b.status}")
            break
        if b.status not in printed_status:
            print(f"[batch] status = {b.status}")
            printed_status.add(b.status)

    output_file_id = getattr(b, "output_file_id", None) or (b.output_file_ids[0] if getattr(b, "output_file_ids", None) else None)
    if not output_file_id:
        raise RuntimeError("No output_file_id on completed batch")
    blob = client.files.content(output_file_id)
    data = blob.read() if hasattr(blob, "read") else blob
    with open(NANO_BATCH_OUT_RAW, "wb") as w:
        w.write(data)
    print(f"[nano] batch output -> {NANO_BATCH_OUT_RAW}")

    qmap = {}
    with open(FIVEHUNDREDQUESTION, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            qmap[ex["id"]] = (ex["question"], ex["gold"])

    preds = []
with open(NANO_BATCH_OUT_RAW, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        cid = obj.get("custom_id","")
        if not cid.startswith("g5nano::"):
            continue
        qid = cid.split("g5nano::",1)[1]
        body = (obj.get("response") or {}).get("body") or {}
        pred = ""
        out = body.get("output", None)
        if isinstance(out, list):
            texts = []
            for piece in out:
                if piece.get("type") == "message":
                    for c in piece.get("content", []):
                        t = c.get("text")
                        if isinstance(t, str):
                            texts.append(t.strip())
            if texts:
                pred = " ".join(t for t in texts if t)

        if not pred and isinstance(body.get("output_text"), str):
            pred = body["output_text"].strip()

        if not pred and isinstance(out, dict):
            if isinstance(out.get("text"), str):
                pred = out["text"].strip()
            elif isinstance(out.get("choices"), list) and out["choices"]:
                msg = out["choices"][0].get("message", {})
                content = msg.get("content")
                if isinstance(content, str):
                    pred = content.strip()
                elif isinstance(content, list):
                    texts = [p.get("text","") for p in content if isinstance(p, dict)]
                    pred = " ".join(t.strip() for t in texts if t.strip())

        q, g = qmap.get(qid, ("",""))
        preds.append({"id": qid, "question": q, "gold": g, "pred": pred})

    with open(NANO_PREDS, "w", encoding="utf-8") as w:
        for r in preds:
            w.write(json.dumps(r) + "\n")
    print(f"[nano] parsed predictions -> {NANO_PREDS}")

    if not os.path.exists(FIVEHUNDREDQUESTION):
        raise FileNotFoundError(f"Missing {FIVEHUNDREDQUESTION}. Uncomment the extract block once to create it.")

    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("Missing OPENROUTER_API_KEY in environment or .env")

    def call_openrouter(question: str) -> str:
        openrouter = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        try:
            chunks = get_sources(col, question, k=5)
            rag_user = make_user_prompt(question, chunks)
            res = openrouter.chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_RAG},
                    {"role": "user", "content": rag_user}
                ],
                temperature=0.2,
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            print("[openrouter error]", e)
            return "[error]"

    with open(QWEN_PREDS, "w", encoding="utf-8") as w, open(FIVEHUNDREDQUESTION, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            ex = json.loads(line)
            pred = call_openrouter(ex["question"])
            w.write(json.dumps({"id": ex["id"], "question": ex["question"], "gold": ex["gold"], "pred": pred}) + "\n")
            print(f"Processing question {i}: {ex['question']}")
            time.sleep(.25)
            if i % 50 == 0:
                print(f"[qwen] {i}/500")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in environment or .env")
    client = OpenAI()

    # judge schema already defined above (score + explanation)

    def write_judge_batch(src_preds_path: str, out_path: str, limit: int | None = None):
        n = 0
        with open(src_preds_path, "r", encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as w:
            for line in f:
                ex = json.loads(line)
                job = {
                    "custom_id": f"judge::{ex['id']}",
                    "method": "POST",
                    "url": ENDPOINT,
                    "body": {
                        "model": GPT5_Mini,
                        "input": [
                            {
                                "role": "user",
                                "content": JUDGE_PROMPT_TMPL.format(
                                    question=ex["question"],
                                    student_response=ex["pred"],
                                    correct_answers=ex["gold"]
                                )
                            }
                        ],
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": "scoring_schema",
                                "schema": judge_schema,
                                "strict": True
                            },
                            "verbosity": "low"
                        },
                    }
                }
                w.write(json.dumps(job) + "\n")
                n += 1
                if limit is not None and n >= limit:
                    break
        print(f"[judge] built -> {out_path} ({n} jobs)")

    # --- Judge qwen first ---
    write_judge_batch(QWEN_PREDS, QWEN_JUDGE_IN, limit=None)
    with open(QWEN_JUDGE_IN, "rb") as fh:
        up = client.files.create(file=fh, purpose="batch")
    b = client.batches.create(input_file_id=up.id,  endpoint= ENDPOINT, completion_window="24h")
    print(f"[judge/batch] id={b.id} status={b.status}")

    printed_status = set()
    while True:
        time.sleep(5)
        b = client.batches.retrieve(b.id)
        if b.status in ("completed","failed","expired","cancelled"):
            print(f"[judge/batch] final status={b.status}")
            if b.status != "completed":
                raise RuntimeError(f"Judge batch ended with {b.status}")
            break
        if b.status not in printed_status:
            print(f"[judge/batch] status={b.status}")
            printed_status.add(b.status)

    ofid = getattr(b, "output_file_id", None) or getattr(b, "error_file_id", None)
    if not ofid:
        raise RuntimeError(
            f"Batch not ready yet. status={b.status} "
            f"completed={getattr(b, 'request_counts', {}).get('completed', '?')}/"
            f"{getattr(b, 'request_counts', {}).get('total', '?')}"
        )

    resp = client.files.content(ofid)
    text = getattr(resp, "text", None)
    if isinstance(text, str):
        with open("batch_output.jsonl", "w", encoding="utf-8") as f:
            f.write(text)
    blob = client.files.content(ofid)
    data = blob.read() if hasattr(blob, "read") else blob
    with open(QWEN_JUDGE_OUT, "wb") as w:
        w.write(data)
    print(f"[judge] saved -> {QWEN_JUDGE_OUT}")

    def parse_judge_map(path: str):
        m = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cid = obj.get("custom_id", "")
                if not cid.startswith("judge::"):
                    continue
                qid = cid.split("judge::", 1)[1]
                body = (obj.get("response") or {}).get("body") or {}
                parsed = None
                out = body.get("output")
                if isinstance(out, dict) and "parsed" in out:
                    parsed = out["parsed"]
                if parsed is None and isinstance(body.get("output_text"), str):
                    try:
                        parsed = json.loads(body["output_text"])
                    except Exception:
                        parsed = None
                val = None
                if isinstance(parsed, dict):
                    if "score" in parsed:
                        val = bool(parsed["score"])
                    elif "correct" in parsed:
                        val = bool(parsed["correct"])
                m[qid] = bool(val)
        return m

    qwen_map = parse_judge_map(QWEN_JUDGE_OUT)
    q_total = len(qwen_map)
    q_correct = sum(1 for v in qwen_map.values() if v)
    print("\nQ2 Prompt (qwen/qwen3-8b)")
    print("System:", SYSTEM_PROMPT_RAG)
    print(f"\nQ4 Total accuracy qwen/qwen3-8b: { (q_correct/q_total if q_total else 0):.3%}  ({q_correct}/{q_total})")

    # --- Judge nano next ---
    if os.path.exists(NANO_PREDS):
        write_judge_batch(NANO_PREDS, NANO_JUDGE_IN, limit=None)
        with open(NANO_JUDGE_IN, "rb") as fh:
            up2 = client.files.create(file=fh, purpose="batch")
        b2 = client.batches.create(input_file_id=up2.id, endpoint=ENDPOINT, completion_window="24h")
        print(f"[judge/batch:nano] id={b2.id} status={b2.status}")

        seen = set()
        while True:
            time.sleep(5)
            b2 = client.batches.retrieve(b2.id)
            if b2.status in ("completed", "failed", "expired", "cancelled"):
                print(f"[judge/batch:nano] final status={b2.status}")
                if b2.status != "completed":
                    raise RuntimeError(f"Nano judge batch ended with {b2.status}")
                break
            if b2.status not in seen:
                print(f"[judge/batch:nano] status={b2.status}")
                seen.add(b2.status)

        ofid2 = getattr(b2, "output_file_id", None) or getattr(b2, "error_file_id", None)
        if not ofid2:
            raise RuntimeError("No output_file_id for nano judge batch")
        blob2 = client.files.content(ofid2)
        data2 = blob2.read() if hasattr(blob2, "read") else blob2
        with open(NANO_JUDGE_OUT, "wb") as w:
            w.write(data2)
        print(f"[judge:nano] saved -> {NANO_JUDGE_OUT}")

        nano_map = parse_judge_map(NANO_JUDGE_OUT)
        n_total = len(nano_map)
        n_correct = sum(1 for v in nano_map.values() if v)
        print(f"\nQ3 Total accuracy GPT-5-nano: {(n_correct / n_total if n_total else 0):.3%}  ({n_correct}/{n_total})")
    else:
        print(f"[nano] skip: predictions not found -> {NANO_PREDS}")

    # (The duplicate judge pass for nano in your original code is removed for cleanliness)
