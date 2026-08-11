# chatgpt_like_agent.py
# Chat-style console assistant (OpenAI + Pro Web + Persistent Memory)
# Web: async httpx (con fallback stdlib), robots.txt, rate limit por dominio, búsqueda (duckduckgo_search),
#      extracción de contenido legible (readability-lxml), resumen de URLs.
# Otras herramientas: hora local, cálculo seguro, I/O en sandbox, memoria persistente.

from __future__ import annotations
import os, re, json, math, time, ast, html, urllib.request, urllib.parse, urllib.error, urllib.robotparser
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# ================= Utilities =================
def normalize_text(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", s.lower())

def clamp(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n-3] + "..."

def bow(text: str) -> Dict[str, float]:
    from collections import Counter
    c = Counter(normalize_text(text))
    if not c: return {}
    norm = math.sqrt(sum(v*v for v in c.values()))
    return {k: v/norm for k, v in c.items()} if norm else dict(c)

def cosine(a: Dict[str,float], b: Dict[str,float]) -> float:
    if not a or not b: return 0.0
    common = set(a)&set(b)
    num = sum(a[t]*b[t] for t in common)
    da = math.sqrt(sum(v*v for v in a.values()))
    db = math.sqrt(sum(v*v for v in b.values()))
    return (num/(da*db)) if da and db else 0.0

def top_k_sentences(text: str, k=6, min_len=50) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if len(s.strip())>=2]
    if not sents: return []
    corp = bow(text); scored=[]
    for s in sents:
        score = sum(corp.get(tok,0.0) for tok in normalize_text(s))
        scored.append((score,s))
    scored.sort(key=lambda x:x[0], reverse=True)
    out=[]
    for _,s in scored:
        if len(s)>=min_len: out.append(s)
        if len(out)>=k: break
    return out or sents[:min(k, len(sents))]

# ================= Memory (persistent) =================
@dataclass
class MemItem:
    role: str  # "user" | "assistant" | "memory"
    text: str
    vec: Dict[str,float] = field(default_factory=dict)

class ConversationMemory:
    def __init__(self, path="memory.json", max_turns=24, max_archive=1000):
        self.path=path; self.max_turns=max_turns; self.max_archive=max_archive
        self.turns: List[MemItem]=[]; self.archive: List[MemItem]=[]
        self._load()

    def _load(self):
        if not os.path.exists(self.path): return
        try:
            data=json.load(open(self.path,"r",encoding="utf-8"))
            self.turns=[MemItem(**d) for d in data.get("turns",[])]
            self.archive=[MemItem(**d) for d in data.get("archive",[])]
        except Exception:
            self.turns, self.archive = [], []

    def save(self):
        json.dump({
            "turns":[t.__dict__ for t in self.turns][-self.max_turns:],
            "archive":[a.__dict__ for a in self.archive][-self.max_archive:]
        }, open(self.path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

    def add(self, role: str, text: str):
        it=MemItem(role=role,text=text,vec=bow(text)); self.turns.append(it)
        if len(self.turns)>self.max_turns: self.archive.append(self.turns.pop(0))
        self.save()

    def add_memory(self, text: str):
        it=MemItem(role="memory", text=text, vec=bow(text))
        self.archive.append(it)
        self.turns.append(it)
        if len(self.turns)>self.max_turns: self.archive.append(self.turns.pop(0))
        self.save()

    def retrieve(self, q: str, k=6) -> List[MemItem]:
        qv=bow(q); pool=self.archive+self.turns
        scored=[(cosine(qv,it.vec),it) for it in pool]; scored.sort(key=lambda x:x[0], reverse=True)
        return [it for s,it in scored[:k] if s>0.05]

    def list_memories(self, k=10) -> List[str]:
        items=[it.text for it in (self.archive+self.turns) if it.role=="memory"]
        return items[-k:] if items else []

# ================= Safe calculator =================
import math as _math
ALLOWED_AST={ast.Expression,ast.Module,ast.Expr,ast.BinOp,ast.UnaryOp,ast.Load,
             ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.Mod,ast.FloorDiv,ast.USub,ast.UAdd,
             ast.Call,ast.Name,ast.Attribute,ast.Constant,ast.Tuple,ast.List}
SAFE_MATH={k:getattr(_math,k) for k in dir(_math) if not k.startswith("_")}
SAFE_MATH.update({"abs":abs,"round":round,"min":min,"max":max})

def safe_eval(expr: str):
    tree=ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_AST: raise ValueError(f"Disallowed: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_MATH: raise ValueError("Only math functions allowed")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in SAFE_MATH: raise ValueError("Only math functions allowed")
            else: raise ValueError("Only simple calls")
        if isinstance(node, ast.Name) and node.id not in SAFE_MATH: raise ValueError("Only math names allowed")
    return eval(compile(tree,"<safe>","eval"), {"__builtins__":{}}, SAFE_MATH)

# ================= Pro Web Client =================
USER_AGENT="Mozilla/5.0 (ChatLikeAgent/1.0; +local-agent)"
MAX_BYTES=2_500_000
DEFAULT_TIMEOUT=18
PER_HOST_DELAY=1.0  # segundos entre peticiones por host (evita abusos)

def _is_http(url: str) -> bool:
    return bool(re.match(r"^https?://", url, re.I))

# --- stdlib fallback ---
def _http_get_stdlib(url: str) -> Tuple[int, Dict[str,str], bytes]:
    req=urllib.request.Request(url, headers={"User-Agent":USER_AGENT,"Accept":"*/*"})
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT) as resp:
        headers={k.lower():v for k,v in resp.headers.items()}
        status=getattr(resp,"status",200)
        chunks=[]; total=0
        while True:
            b=resp.read(65536)
            if not b: break
            chunks.append(b); total+=len(b)
            if total>MAX_BYTES: break
        return status, headers, b"".join(chunks)

def _charset(headers: Dict[str,str], default="utf-8") -> str:
    ct=headers.get("content-type","")
    m=re.search(r"charset=([A-Za-z0-9_\-]+)", ct)
    return m.group(1) if m else default

def _html_to_text(html_str: str) -> str:
    html_str=re.sub(r"(?is)<script.*?>.*?</script>"," ",html_str)
    html_str=re.sub(r"(?is)<style.*?>.*?</style>"," ",html_str)
    txt=re.sub(r"(?is)<[^>]+>"," ",html_str)
    txt=html.unescape(txt)
    txt=re.sub(r"\s+"," ",txt).strip()
    return txt

# --- optional deps ---
try:
    import httpx
except Exception:
    httpx=None

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS=None

try:
    from readability import Document as ReadabilityDocument
    from bs4 import BeautifulSoup
except Exception:
    ReadabilityDocument=None
    BeautifulSoup=None

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text=None

_last_host_hit: Dict[str, float] = {}

def _respect_rate_limit(host: str):
    now=time.time()
    last=_last_host_hit.get(host, 0.0)
    delta=now-last
    if delta < PER_HOST_DELAY:
        time.sleep(PER_HOST_DELAY - delta)
    _last_host_hit[host]=time.time()

def robots_allowed(url: str) -> bool:
    try:
        parsed=urllib.parse.urlparse(url)
        robots_url=f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp=urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        _respect_rate_limit(parsed.netloc)
        try:
            rp.read()
        except Exception:
            # si robots falla, asumimos permitido pero seguimos siendo amables
            return True
        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        return True

def http_get(url: str) -> Tuple[int, Dict[str,str], bytes]:
    if not _is_http(url):
        raise ValueError("Invalid URL")
    if not robots_allowed(url):
        return 999, {"content-type":"text/plain"}, b"[Blocked by robots.txt]"
    parsed=urllib.parse.urlparse(url)
    _respect_rate_limit(parsed.netloc)
    # httpx if available
    if httpx is not None:
        try:
            with httpx.Client(http2=True, headers={"User-Agent":USER_AGENT}, timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
                r=client.get(url)
                data=r.content[:MAX_BYTES]
                headers=dict(r.headers)
                return r.status_code, {k.lower():v for k,v in headers.items()}, data
        except Exception:
            pass
    # fallback
    return _http_get_stdlib(url)

def extract_text_from_response(headers: Dict[str,str], data: bytes) -> str:
    ctype=headers.get("content-type","").lower()
    # PDF
    if "application/pdf" in ctype and pdf_extract_text is not None:
        try:
            import io
            text=pdf_extract_text(io.BytesIO(data))
            return clamp(text, 16000) or "(empty pdf)"
        except Exception:
            return "(failed to parse pdf; install pdfminer.six)"
    # HTML/TEXT
    if "html" in ctype or "text" in ctype or ctype=="":
        try:
            txt=data.decode(_charset(headers), errors="ignore")
        except LookupError:
            txt=data.decode("utf-8", errors="ignore")
        # readability if available
        if ReadabilityDocument is not None and BeautifulSoup is not None:
            try:
                doc=ReadabilityDocument(txt)
                article_html=doc.summary(html_partial=True)
                soup=BeautifulSoup(article_html, "lxml")
                cleaned=soup.get_text(" ", strip=True)
                return clamp(cleaned, 16000) or "(empty)"
            except Exception:
                pass
        return clamp(_html_to_text(txt), 16000) or "(empty)"
    return f"(binary {len(data)} bytes; {ctype})"

def web_preview(url: str, short=True) -> str:
    try:
        st,h,data=http_get(url)
    except Exception as e:
        return f"HTTP error: {e}"
    if st==999:
        return "Blocked by robots.txt (site does not allow automated fetch)."
    text=extract_text_from_response(h, data)
    return clamp(text, 1200 if short else 4000)

def web_summarize(url: str) -> str:
    content=web_preview(url, short=False)
    if content.startswith("HTTP error") or content.startswith("Blocked by robots") or content.startswith("(binary"):
        return content
    bullets=top_k_sentences(content, k=6, min_len=60)
    return "\n".join(f"- {b}" for b in bullets) or clamp(content,800)

def web_search(query: str, n: int = 6) -> List[Tuple[str,str]]:
    if DDGS is not None:
        try:
            with DDGS() as ddgs:
                results=[]
                for r in ddgs.text(query, max_results=n):
                    # r: {'title':..., 'href':..., 'body':...}
                    if "href" in r and "title" in r:
                        results.append((r["title"], r["href"]))
                    if len(results)>=n: break
                return results
        except Exception:
            pass
    # fallback: muy básico via html de duckduckgo
    base = "https://duckduckgo.com/html/"
    q = urllib.parse.urlencode({"q": query})
    url = f"{base}?{q}"
    html_text = web_preview(url, short=False)
    links = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html_text, flags=re.I)
    results = []
    for href, title in links:
        title = re.sub("<.*?>", "", title)
        href = html.unescape(href)
        if href.startswith("/"):  # redireccion interna
            continue
        results.append((title.strip(), href.strip()))
        if len(results) >= n: break
    return results

# ================= OpenAI integration =================
try:
    from openai import OpenAI
except Exception:
    OpenAI=None

OPENAI_SYSTEM_PROMPT = (
    "You are the assistant of a console app with INTERNET ACCESS. "
    "Tools available: live web fetch (respects robots.txt, per-host rate-limit), "
    "basic web search (DuckDuckGo), readability extraction, local time, safe math, file I/O, and persistent memory. "
    "If asked about internet or recency, say YES you can fetch pages and search now. "
    "Offer to open/summary URLs or search queries. Be concise and practical."
)

def call_llm_openai(history: List[Dict[str,str]], user: str, model="gpt-4o-mini") -> str:
    if OpenAI is None: return "[OpenAI SDK not installed. Run: pip install openai]"
    api_key=os.getenv("OPENAI_API_KEY")
    if not api_key:
    return "[OpenAI API key not configured. Local tools are still available.]"
    client=OpenAI(api_key=api_key)

    msgs=[{"role":"system","content":OPENAI_SYSTEM_PROMPT}]
    msgs.extend({"role":h["role"],"content":h["content"]} for h in history[-10:])
    msgs.append({"role":"user","content":user})
    try:
        resp=client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0.7,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI error] {e}"

# ================= Agent =================
WORKSPACE="workspace"; os.makedirs(WORKSPACE, exist_ok=True)

class ChatLikeAgent:
    def __init__(self): self.mem=ConversationMemory()

    # ---- Memory intents ----
    def _do_remember(self, text: str) -> Optional[str]:
        m = re.search(r"(?:recuerda(?: que)?|guarda en memoria|memoriza|remember(?: that)?)[:\s]+(.+)", text, re.I|re.S)
        if not m: return None
        fact = m.group(1).strip()
        if not fact: return "Necesito qué quieres que recuerde."
        self.mem.add_memory(fact)
        return f"Lo recordaré: {fact}"

    def _do_recall(self, text: str) -> Optional[str]:
        if re.search(r"\b(qué\s+recuerdas|que\s+recuerdas|recall|memoria|what\s+do\s+you\s+remember)\b", text, re.I):
            m = re.search(r"(?:de|about)\s+(.+)$", text, re.I)
            if m:
                q = m.group(1).strip()
                hits = self.mem.retrieve(q, k=8)
                if not hits: return f"No tengo recuerdos sobre: {q}"
                lines = [f"- {it.text}" for it in hits if it.role == "memory"]
                return "\n".join(lines) if lines else f"No tengo recuerdos sobre: {q}"
            items = self.mem.list_memories(k=10)
            return "Recuerdos:\n" + "\n".join(f"- {t}" for t in items) if items else "Aún no tengo recuerdos guardados."
        return None

    # ---- Internet/Q&A intents ----
    def _do_internet_intent(self, text: str) -> Optional[str]:
        if re.search(r"(tienes\s+acceso\s+al?\s+internet|acceso\s+a\s+internet|internet\s+en\s+tiempo\s+real|online|conectado)", text, re.I):
            return ("Sí: puedo abrir páginas y hacer búsquedas al momento. "
                    "Prueba: 'Resume esta página: https://www.python.org/' o 'Busca últimas noticias de Python'.")
        return None

    def _do_time_intent(self, text: str) -> Optional[str]:
        if re.search(r"\b(hora|time|date|fecha)\b", text, re.I):
            return self._do_time()
        return None

    def _do_math(self, text: str) -> Optional[str]:
        if re.search(r"[\d\)\]]\s*[\+\-\*\/\^]|sqrt|sin|cos|tan|pi|\b e\b", text, re.I):
            try:
                expr=text.replace("^","**")
                return f"{text} = {safe_eval(expr)}"
            except Exception:
                return None
        return None

    def _do_time(self) -> str:
        return time.strftime("Current time: %Y-%m-%d %H:%M:%S", time.localtime())

    def _do_write(self, text: str) -> Optional[str]:
        m=re.search(r"(?:save|write|guardar|guarda)\s+['\"]?([A-Za-z0-9._\-\/]+)['\"]?\s+(?:with|con|:)\s+(.+)$", text, re.I|re.S)
        if not m: return None
        fname,content=m.group(1),m.group(2).strip()
        path=os.path.normpath(os.path.join(WORKSPACE,fname))
        if not os.path.abspath(path).startswith(os.path.abspath(WORKSPACE)): return "Refused: path must be inside workspace/."
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path,"w",encoding="utf-8").write(content)
        return f"Saved to workspace/{fname}"

    def _do_read(self, text: str) -> Optional[str]:
        m=re.search(r"(?:read|open|lee|abrir)\s+['\"]?([A-Za-z0-9._\-\/]+)['\"]?", text, re.I)
        if not m: return None
        fname=m.group(1); path=os.path.normpath(os.path.join(WORKSPACE,fname))
        if not os.path.abspath(path).startswith(os.path.abspath(WORKSPACE)): return "Refused: path must be inside workspace/."
        if not os.path.exists(path): return f"File not found: {fname}"
        return f"[workspace/{fname}]\n"+open(path,"r",encoding="utf-8").read()

    def _do_web(self, text: str) -> Optional[str]:
        # URL directa
        m=re.search(r"(https?://\S+)", text, re.I)
        if m:
            url=m.group(1).rstrip(").,]")
            if any(k in text.lower() for k in ["resume","resumen","summarize","summary"]):
                return f"Summary of {url}:\n{web_summarize(url)}"
            return f"Preview of {url}:\n{web_preview(url, short=True)}"
        # búsqueda
        if re.search(r"\b(busca|buscar|búscame|search|investiga|noticias|últimas|ultimas|hoy)\b", text, re.I):
            q = re.sub(r"(?i)\b(busca|buscar|búscame|search|investiga)\b", "", text).strip(" :,-")
            if len(q)<3: q = text
            hits = web_search(q, n=6)
            if not hits: return "No encontré resultados."
            lines = [f"{i+1}. {t} — {u}" for i,(t,u) in enumerate(hits)]
            return "Resultados web:\n" + "\n".join(lines)
        return None

    # ---- Heuristic / LLM ----
    def _heuristic_answer(self, user: str) -> str:
        # 1) Intención de hora/fecha: responde localmente
        if re.search(r"\b(hora|time|date|fecha)\b", user, re.I):
            return self._do_time()

        # 2) Probar OpenAI
        history = [{"role": it.role, "content": it.text} for it in self.mem.turns]
        reply = call_llm_openai(history, user, model="gpt-4o-mini")
        if reply and not reply.startswith("[OpenAI error]") and not reply.startswith("[OpenAI SDK") and not reply.startswith("[Set OPENAI_API_KEY"):
            return reply

        # 3) Fallback
        return ("Tell me more about your goal, or paste a link/data and I'll analyze it. "
                "I can search pages, summarize, do safe math, and manage files locally.")

    def chat(self, user: str) -> str:
        self.mem.add("user", user)
        # Planner: memoria → internet-intent → hora → archivos → web → cálculo → LLM
        for fn in (self._do_remember, self._do_recall, self._do_internet_intent,
                   self._do_time_intent, self._do_write, self._do_read, self._do_web, self._do_math):
            out = fn(user)
            if out:
                self.mem.add("assistant", out)
                return out
        out=self._heuristic_answer(user)
        self.mem.add("assistant", out)
        return out

# ================= CLI =================
def main():
    print("Chat-like Agent (OpenAI + PRO Web + Persistent Memory). Type 'exit' to quit.")
    agent=ChatLikeAgent()
    while True:
        try:
            user=input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAssistant: Bye!"); break
        if not user: continue
        if user.lower() in ("exit","quit","salir"):
            print("Assistant: Bye!"); break
        print("Assistant:", agent.chat(user))

if __name__=="__main__":
    main()
