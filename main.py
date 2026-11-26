"""
스타트업 투자유치 뉴스 자동 브리핑

1. 구글 뉴스 RSS에서 최근 투자유치 기사를 모은다.
2. Gemini가 기사 제목·요약에서 기업명, 업종, 투자 단계, 금액, 투자사를 뽑는다.
   (기사에 없는 내용은 비워두도록 지시하고, 결과는 다시 한 번 기사 원문 제목과 대조한다)
3. 텔레그램으로 브리핑을 보내고, data/deals.csv와 data/briefings/날짜.md에 기록한다.
4. (선택) 노션 데이터베이스에 한 건씩 저장한다.

사용법:
  python main.py            # 최근 24시간 기사
  python main.py --hours 72 # 최근 72시간 기사
  python main.py --dry-run  # 전송/저장 없이 결과만 출력
"""

import argparse
import csv
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import requests

KST = timezone(timedelta(hours=9))
DATA_DIR = Path(__file__).parent / "data"
QUERIES = [
    "스타트업 투자 유치",
    "시리즈A 투자 유치",
    "시리즈B 투자 유치",
    "시드 투자 유치",
    "프리A 투자 유치",
]
# 앞의 모델이 바쁘거나(503) 없으면(404) 다음 모델로 자동 전환
GEMINI_MODELS = [m for m in [
    (os.environ.get("GEMINI_MODEL") or "").strip(),
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
] if m]
BATCH_SIZE = 40
# 제목에 이 단어가 하나도 없으면 투자유치 기사가 아니라고 보고 AI에 보내지 않는다.
FUNDING_WORDS = ["투자", "유치", "시리즈", "시드", "펀딩"]


def env(name):
    """Secrets에 실수로 들어간 공백·줄바꿈을 지운다."""
    return (os.environ.get(name) or "").strip()


# ---------- 1. 뉴스 수집 ----------

def fetch_rss(query):
    url = (f"https://news.google.com/rss/search?q={quote(query)}+when:2d"
           "&hl=ko&gl=KR&ceid=KR:ko")
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return resp.text


def parse_rss(xml_text):
    items = []
    root = ET.fromstring(xml_text)
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].strip()
        pub = item.findtext("pubDate")
        items.append({
            "title": title,
            "link": (item.findtext("link") or "").strip(),
            "source": source,
            "published": parsedate_to_datetime(pub).astimezone(KST) if pub else None,
        })
    return items


def normalize(title):
    return re.sub(r"[^0-9A-Za-z가-힣]", "", title)[:30]


def collect_news(hours):
    cutoff = datetime.now(KST) - timedelta(hours=hours)
    seen, news = set(), []
    for q in QUERIES:
        try:
            items = parse_rss(fetch_rss(q))
        except Exception as e:  # 한 검색어가 실패해도 나머지는 계속
            print(f"[경고] '{q}' 수집 실패: {e}")
            continue
        for it in items:
            if not any(w in it["title"] for w in FUNDING_WORDS):
                continue
            key = normalize(it["title"])
            if key in seen or (it["published"] and it["published"] < cutoff):
                continue
            seen.add(key)
            news.append(it)
        time.sleep(1)
    news.sort(key=lambda x: x["published"] or cutoff, reverse=True)
    return news


# ---------- 2. AI 정보 추출 ----------

PROMPT = """너는 벤처캐피털의 리서치 보조다. 아래 뉴스 제목 목록에서 '스타트업이 투자를 유치했다'는 기사만 골라
각 기사에 대해 다음 정보를 JSON 배열로 돌려줘.

규칙:
- 기사 제목에 근거가 없는 내용은 절대 추측하지 말고 null로 둔다.
- 투자유치 기사가 아니면(행사, 펀드 결성, 정책, 칼럼, 상장 등) 결과에서 제외한다.
- 같은 기업의 같은 투자 건이 여러 번 나오면 하나만 남긴다.
- amount_krw_100m: 투자 금액을 '억원' 단위 숫자로. 달러는 1달러=1,400원으로 환산하고 converted=true로 표시.
- stage: "시드", "프리A", "시리즈A", "시리즈B", "시리즈C 이상", "브릿지", "전략적 투자" 중 하나 또는 null.

출력 형식 (JSON 배열만 출력):
[{{"id": 기사번호, "company": "기업명", "sector": "업종(짧게)", "stage": ..., "amount_krw_100m": 숫자 또는 null,
   "converted": true/false, "investors": "투자사(쉼표 구분)" 또는 null, "summary": "한 줄 요약"}}]

뉴스 목록:
{items}
"""


_working_model = None  # 한 번 성공한 모델은 다음 배치에서도 먼저 쓴다
_model_list = None
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def available_models(api_key):
    """이 API 키로 지금 쓸 수 있는 Flash 계열 모델을 구글에 직접 물어본다."""
    global _model_list
    if _model_list is not None:
        return _model_list
    found = []
    try:
        resp = requests.get(f"{GEMINI_BASE}/models", params={"key": api_key, "pageSize": 200}, timeout=30)
        if resp.status_code >= 400:
            print(f"[경고] 모델 목록 조회 실패: {resp.status_code} {resp.text[:200]}")
        else:
            for m in resp.json().get("models", []):
                name = m.get("name", "").replace("models/", "")
                methods = m.get("supportedGenerationMethods", [])
                if ("generateContent" in methods and "flash" in name
                        and not any(x in name for x in ("image", "tts", "audio", "live", "embedding", "exp"))):
                    found.append(name)
    except requests.RequestException as e:
        print(f"[경고] 모델 목록 조회 실패: {e}")
    # 미리 정한 순서를 먼저, 그다음 목록에 있는 나머지 Flash 모델 (최신 이름 순)
    preferred = [m for m in GEMINI_MODELS if not found or m in found]
    rest = sorted((m for m in found if m not in preferred), reverse=True)
    _model_list = preferred + rest
    print(f"사용 가능한 모델 후보: {', '.join(_model_list[:6])}")
    return _model_list


def call_gemini(prompt, api_key):
    global _working_model
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    candidates = available_models(api_key)
    models = ([_working_model] if _working_model else []) + \
             [m for m in candidates if m != _working_model]
    errors = []
    for model in models[:6]:
        url = f"{GEMINI_BASE}/models/{model}:generateContent"
        for attempt in range(2):
            try:
                resp = requests.post(url, params={"key": api_key}, json=body, timeout=90)
            except requests.RequestException as e:  # 응답 지연 등
                errors.append(f"{model}: {type(e).__name__}")
                time.sleep(5)
                continue
            if resp.status_code == 404:  # 모델 이름이 없어졌으면 바로 다음 모델로
                errors.append(f"{model}: 404")
                break
            if resp.status_code in (429, 500, 503):
                errors.append(f"{model}: {resp.status_code}")
                time.sleep(8 * (attempt + 1))
                continue
            if resp.status_code >= 400:  # 400/403 = 키 문제일 가능성
                raise RuntimeError(f"{model}: {resp.status_code} {resp.text[:200]}")
            data = resp.json()
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                errors.append(f"{model}: 빈 응답")
                break
            if _working_model != model:
                print(f"사용 모델: {model}")
            _working_model = model
            return json.loads(text)
    raise RuntimeError("모든 모델 실패 → " + ", ".join(errors))


def extract_deals(news, api_key):
    deals, failed, total = [], 0, 0
    for start in range(0, len(news), BATCH_SIZE):
        total += 1
        batch = news[start:start + BATCH_SIZE]
        listing = "\n".join(f"{start + i}. {n['title']} ({n['source']})" for i, n in enumerate(batch))
        try:
            results = call_gemini(PROMPT.format(items=listing), api_key)
        except Exception as e:
            print(f"[경고] AI 추출 실패: {e}")
            failed += 1
            fallback = rule_extract(batch)
            print(f"  → 예비 모드(규칙 추출)로 {len(fallback)}건 정리")
            deals.extend(fallback)
            continue
        if isinstance(results, dict):  # 가끔 {"deals": [...]} 형태로 돌려줄 때
            results = next((v for v in results.values() if isinstance(v, list)), [])
        print(f"  AI 응답 {len(results) if isinstance(results, list) else 0}건 (기사 {start}~{start + len(batch) - 1}번)")
        for r in results if isinstance(results, list) else []:
            if not isinstance(r, dict):
                continue
            try:  # AI가 번호를 "3", 3.0처럼 돌려줘도 받아들인다
                idx = int(float(str(r.get("id")).strip()))
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(news)):
                continue
            article = news[idx]
            try:  # 금액이 "50", "50억" 같은 글자로 와도 숫자로 정리
                amt = r.get("amount_krw_100m")
                r["amount_krw_100m"] = float(re.sub(r"[^0-9.]", "", str(amt))) if amt not in (None, "", "null") else None
            except ValueError:
                r["amount_krw_100m"] = None
            r["title"] = article["title"]
            r["link"] = article["link"]
            r["source"] = article["source"]
            r["date"] = (article["published"] or datetime.now(KST)).strftime("%Y-%m-%d")
            r["check"] = verify(r)
            deals.append(r)
    return dedupe(deals), failed, total


STAGE_PATTERNS = [
    ("프리시리즈A", "프리A"), ("프리A", "프리A"), ("프리 A", "프리A"), ("프리시드", "시드"), ("시드", "시드"),
    ("시리즈A", "시리즈A"), ("시리즈 A", "시리즈A"), ("시리즈B", "시리즈B"), ("시리즈 B", "시리즈B"),
    ("시리즈C", "시리즈C 이상"), ("시리즈D", "시리즈C 이상"), ("브릿지", "브릿지"), ("전략적 투자", "전략적 투자"),
]
AMOUNT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(조|억|만\s*달러|달러|만\s*유로|유로)")
# 투자유치가 아닌 기사에 자주 나오는 단어
NOT_DEAL_WORDS = ["센터", "지원사업", "모집", "공고", "펀드 결성", "출자", "세미나", "데모데이", "설명회", "프로그램"]


def rule_extract(batch):
    """AI가 실패했을 때 쓰는 예비 모드: 제목에서 단계·금액·기업명을 규칙으로 뽑는다."""
    results = []
    for n in batch:
        t = n["title"]
        if "유치" not in t or any(w in t for w in NOT_DEAL_WORDS):
            continue
        stage = next((s for p, s in STAGE_PATTERNS if p in t), None)
        amount, converted = None, False
        m = AMOUNT_RE.search(t)
        if m:
            num, unit = float(m.group(1).replace(",", "")), m.group(2).replace(" ", "")
            amount = {"조": num * 10000, "억": num, "만달러": num * 0.14, "달러": num * 1400 / 1e8,
                      "만유로": num * 0.15, "유로": num * 1500 / 1e8}[unit]
            converted = "달러" in unit or "유로" in unit
        if not stage and amount is None:
            continue
        # 보통 '기업명, ...' 또는 '기업명 ... 유치' 형태라 첫 쉼표 앞을 기업명으로 본다
        head = re.split(r"[,，…·]", t)[0]
        head = re.sub(r"^\s*(\[[^\]]*\]|'[^']*'|\"[^\"]*\"|‘[^’]*’)\s*", "", head).strip()
        words = head.split()
        company = words[-1] if words else None
        results.append({
            "company": company, "sector": None, "stage": stage,
            "amount_krw_100m": round(amount, 1) if amount else None, "converted": converted,
            "investors": None, "summary": "(예비 모드: AI 없이 제목에서 자동 추출)",
            "title": t, "link": n["link"], "source": n["source"],
            "date": (n["published"] or datetime.now(KST)).strftime("%Y-%m-%d"),
        })
    for r in results:
        r["check"] = "규칙추출" if verify(r) == "OK" else "확인필요"
    return results


def verify(deal):
    """AI 결과를 기사 제목과 다시 대조한다. 제목에 기업명이 없으면 '확인필요'."""
    company = (deal.get("company") or "").replace(" ", "")
    title = deal["title"].replace(" ", "")
    if not company or company not in title:
        return "확인필요"
    if deal.get("amount_krw_100m") and not re.search(r"\d", deal["title"]):
        return "금액확인"
    return "OK"


def dedupe(deals):
    # 같은 기사에서 여러 결과가 나오면 검증을 통과한 것 하나만 남긴다.
    by_link = {}
    for d in deals:
        prev = by_link.get(d["link"])
        if prev is None or (prev["check"] != "OK" and d["check"] == "OK"):
            by_link[d["link"]] = d
    best = {}
    for d in by_link.values():
        key = (d.get("company") or d["title"]).replace(" ", "")
        if key not in best or (d.get("amount_krw_100m") and not best[key].get("amount_krw_100m")):
            best[key] = d
    return sorted(best.values(), key=lambda d: d.get("amount_krw_100m") or 0, reverse=True)


# ---------- 3. 출력 ----------

def fmt_amount(d):
    amt = d.get("amount_krw_100m")
    if not amt:
        return "비공개"
    text = f"{amt:,.0f}억원" if amt >= 1 else f"{amt * 100:,.0f}백만원"
    return text + (" (환산)" if d.get("converted") else "")


def format_message(deals, today):
    head = f"🚀 <b>스타트업 투자유치 브리핑</b> ({today})\n"
    if not deals:
        return head + "\n오늘 새로 확인된 투자유치 기사가 없습니다."
    total = sum(d.get("amount_krw_100m") or 0 for d in deals)
    stages = {}
    for d in deals:
        stages[d.get("stage") or "미상"] = stages.get(d.get("stage") or "미상", 0) + 1
    stage_text = ", ".join(f"{k} {v}건" for k, v in stages.items())
    lines = [head, f"총 {len(deals)}건 · 공개 금액 합계 약 {total:,.0f}억원\n({stage_text})\n"]
    for d in deals:
        flag = "" if d["check"] == "OK" else f" ⚠️{d['check']}"
        lines.append(
            f"<b>{d.get('company') or '기업명 미상'}</b> · {d.get('sector') or '-'}{flag}\n"
            f"{d.get('stage') or '단계 미상'} · {fmt_amount(d)}\n"
            f"투자: {d.get('investors') or '-'}\n"
            f"{d.get('summary') or ''}\n"
            f"<a href=\"{d['link']}\">{d['source'] or '기사'}</a>\n"
        )
    return "\n".join(lines)


def send_telegram(text, token, chat_id):
    chunks, current = [], ""
    for block in text.split("\n\n"):
        if len(current) + len(block) > 3800:
            chunks.append(current)
            current = ""
        current += block + "\n\n"
    chunks.append(current)
    for chunk in chunks:
        if chunk.strip():
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML",
                      "disable_web_page_preview": "true"},
                timeout=30,
            ).raise_for_status()


CSV_FIELDS = ["date", "company", "sector", "stage", "amount_krw_100m", "converted",
              "investors", "summary", "check", "source", "title", "link"]


def save_files(deals, message, today):
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / "deals.csv"
    existing = set()
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig") as f:
            existing = {(r["company"], r["date"]) for r in csv.DictReader(f)}
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for d in deals:
            if (str(d.get("company")), d["date"]) not in existing:
                w.writerow(d)

    md_dir = DATA_DIR / "briefings"
    md_dir.mkdir(exist_ok=True)
    md = re.sub(r"<a href=\"(.*?)\">(.*?)</a>", r"[\2](\1)", message)
    md = md.replace("<b>", "**").replace("</b>", "**")
    (md_dir / f"{today}.md").write_text(md, encoding="utf-8")


def save_notion(deals, token, database_id):
    """노션 DB 속성 이름: 기업명(제목), 업종, 단계, 금액(억원), 투자사, 요약, 검증, 날짜, 링크"""
    headers = {"Authorization": f"Bearer {token}", "Notion-Version": "2022-06-28",
               "Content-Type": "application/json"}

    def text(v):
        return {"rich_text": [{"text": {"content": str(v or "")[:1900]}}]}

    for d in deals:
        props = {
            "기업명": {"title": [{"text": {"content": d.get("company") or d["title"][:50]}}]},
            "업종": text(d.get("sector")),
            "단계": {"select": {"name": d.get("stage") or "미상"}},
            "금액(억원)": {"number": d.get("amount_krw_100m")},
            "투자사": text(d.get("investors")),
            "요약": text(d.get("summary")),
            "검증": {"select": {"name": d["check"]}},
            "날짜": {"date": {"start": d["date"]}},
            "링크": {"url": d["link"]},
        }
        resp = requests.post("https://api.notion.com/v1/pages", headers=headers, timeout=30,
                             json={"parent": {"database_id": database_id}, "properties": props})
        if resp.status_code >= 400:
            print(f"[경고] 노션 저장 실패 ({d.get('company')}): {resp.text[:200]}")
        time.sleep(0.4)


# ---------- 실행 ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = env("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY가 설정되지 않았습니다.")

    today = datetime.now(KST).strftime("%Y-%m-%d")
    news = collect_news(args.hours)
    print(f"수집한 투자 관련 기사 {len(news)}건")
    for n in news[:5]:
        print(f"  예시: {n['title']}")
    deals, failed, total = extract_deals(news, api_key)
    message = format_message(deals, today)
    if failed:
        message += (f"\n\n⚠️ AI 분석 {total}묶음 중 {failed}묶음이 실패해 "
                    "해당 기사는 예비 모드(제목 규칙 추출)로 정리했습니다. 업종·투자사가 비어 있을 수 있습니다.")
    print(message)

    if args.dry_run:
        return
    save_files(deals, message, today)

    token = env("TELEGRAM_TOKEN") or env("TELEGRAM_BOT_TOKEN")
    chat_id = env("CHAT_ID") or env("TELEGRAM_CHAT_ID")
    if token and chat_id:
        try:
            send_telegram(message, token, chat_id)
        except requests.HTTPError as e:
            # 기록은 저장되도록 여기서 멈추지 않는다
            print(f"[오류] 텔레그램 전송 실패: {e.response.status_code} {e.response.text[:200]}")
    else:
        print("[경고] TELEGRAM_TOKEN 또는 CHAT_ID가 비어 있습니다.")

    notion_token, notion_db = env("NOTION_TOKEN"), env("NOTION_DATABASE_ID")
    if notion_token and notion_db:
        save_notion(deals, notion_token, notion_db)


if __name__ == "__main__":
    main()
