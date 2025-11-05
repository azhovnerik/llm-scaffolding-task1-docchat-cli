#!/usr/bin/env python3
"""
DocChat CLI — інтерактивна документація

Ключевые моменты:
- Без LangChain/LangGraph/DSPy (только LiteLLM для вызова LLM через OpenRouter)
- RAG-STRICT: ответы формируются исключительно по документам
- Путь к документам теперь можно задавать:
    1) через CLI флаг:   --docs /path/to/docs
    2) через окружение:   DOCS_DIR=/path/to/docs
   Приоритет: CLI > окружение. Если оба отсутствуют — ошибка.

Модели (≤30B): meta-llama/llama-3.1-8b-instruct, qwen3-30b-instruct, gpt-oss-20b, qwen2.5-14b-instruct (точный id из OpenRouter).

ENV:
  OPENROUTER_API_KEY=...
  (опц.) OPENROUTER_REFERRER=yourdomain.tld
  (опц.) OPENROUTER_APP_NAME=DocChatCLI/1.1
  (опц.) DOCS_DIR=/absolute/path/to/docs
"""
from __future__ import annotations
import argparse
import os
import sys
import re
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

# --- LiteLLM (HTTP-клиент к OpenRouter) ---
from litellm import completion  # type: ignore

# --- Простой RAG-индекс (TF-IDF) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

console = Console()

# ------------------------- Чтение файлов -------------------------

def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        console.print(f"[red]Не удалось прочитать {path}: {e}[/red]")
        return ""


def read_html_file(path: str) -> str:
    html = read_text_file(path)
    try:
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(separator="\n")
    except Exception:
        return html


def read_erb_file(path: str) -> str:
    """
    Чтение .erb (Ruby ERB шаблоны: HTML + <% ... %> вставки).
    1) Удаляем ERB-вставки (<% ... %>, <%= ... %>, <%- ... %>)
    2) Парсим как HTML и извлекаем текст.
    """
    raw = read_text_file(path)
    if not raw:
        return ""
    cleaned = re.sub(r"<%[=\-\s]?.*?%>", " ", raw, flags=re.DOTALL)
    try:
        soup = BeautifulSoup(cleaned, "lxml")
        return soup.get_text(separator="\n")
    except Exception:
        return cleaned


def read_pdf_file(path: str) -> str:
    try:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except Exception as e:
        console.print(f"[yellow]Предупреждение: PDF {path} прочитан частично/пусто: {e}[/yellow]")
        return ""


def iter_docs(root: str) -> Iterable[Tuple[str, str]]:
    """
    Рекурсивно обходит каталог root (неограниченная глубина) и
    возвращает (путь, текст) для поддерживаемых расширений.
    """
    exts = {
        ".txt": read_text_file,
        ".md": read_text_file,
        ".html": read_html_file,
        ".htm": read_html_file,
        ".erb": read_erb_file,
        ".pdf": read_pdf_file,
    }
    ignore_dirs = {".git", "node_modules", "dist", "build", ".venv", "venv", "__pycache__"}

    for dirpath, dirnames, filenames in os.walk(root):
        # фильтруем нежелательные каталоги на лету
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]

        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in exts:
                path = os.path.join(dirpath, name)
                text = exts[ext](path)
                if text.strip():
                    yield path, text


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    # Валидация параметров
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")

    text = re.sub(r"\s+", " ", text).strip()
    n = len(text)
    chunks: list[str] = []
    i = 0
    step = max_chars - overlap  # положительный шаг

    while i < n:
        j = min(i + max_chars, n)
        chunks.append(text[i:j])
        if j == n:  # конец — выходим
            break
        i += step

    return chunks


@dataclass
class RagChunk:
    doc_path: str
    chunk_text: str
    chunk_id: int


class RagIndex:
    def __init__(self):
        self.chunks: List[RagChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None

    def build(self, docs_dir: str, max_chars: int = 2000, overlap: int = 200):
        self.chunks.clear()
        for path, text in iter_docs(docs_dir):
            pieces = chunk_text(text, max_chars=max_chars, overlap=overlap)
            for i, ch in enumerate(pieces):
                self.chunks.append(RagChunk(path, ch, i))
        if not self.chunks:
            console.print("[red]Ошибка: в указанной папке нет поддерживаемых файлов (.txt, .md, .html/.htm, .erb, .pdf) или они пустые.[/red]")
            return
        # Ограничим размер словаря, чтобы экономить память на больших корпусах
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        corpus = [c.chunk_text for c in self.chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)
        console.print(f"[green]✔ Индекс построен: {len(self.chunks)} чанков[/green]")

    def retrieve(self, query: str, top_k: int = 4) -> List[RagChunk]:
        if not self.chunks or self.vectorizer is None or self.matrix is None:
            return []
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]


# --------------------- LLM клиент (OpenRouter) ---------------------

def openrouter_headers() -> dict:
    headers = {}
    ref = os.getenv("OPENROUTER_REFERRER")
    app = os.getenv("OPENROUTER_APP_NAME")
    if ref:
        headers["HTTP-Referer"] = ref
    if app:
        headers["X-Title"] = app
    return headers


def llm_complete(model: str, messages: list[dict], temperature: float = 0.2,
                 max_tokens: int = 800, stream: bool = True) -> str:
    api_base = "https://openrouter.ai/api/v1"
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY не задан[/red]")
        sys.exit(1)

    # Нормализуем модель под OpenRouter: требуем префикс openrouter/
    model_id = model if model.startswith("openrouter/") else f"openrouter/{model}"

    try:
        if stream:
            resp = completion(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                api_key=api_key,
                api_base=api_base,
                extra_headers=openrouter_headers(),
            )
            out = []
            for chunk in resp:
                delta = None
                try:
                    delta = chunk.choices[0].delta.get("content")
                except Exception:
                    pass
                if delta:
                    out.append(delta)
                    console.print(delta, end="")
            console.print()
            return "".join(out)
        else:
            resp = completion(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                api_base=api_base,
                extra_headers=openrouter_headers(),
            )
            return resp.choices[0].message["content"]
    except Exception as e:
        console.print(f"[red]LLM ошибка: {e}[/red]")
        return ""


# --------------------- Подготовка подсказок ---------------------

def build_system_prompt() -> str:
    return (
        "Ви асистент документації в режимі RAG-STRICT. ВІДПОВІДАЙТЕ ВИКЛЮЧНО "
        "на підставі наданих уривків документів. Не робіть припущень і не "
        "додавайте зовнішні знання. Якщо релевантного контексту бракує — відповідь НЕ формувати."
    )


def build_user_prompt(user_query: str, contexts: List[RagChunk]) -> str:
    ctx_texts = []
    for c in contexts:
        header = f"[DOC:{os.path.basename(c.doc_path)}#{c.chunk_id}]\n"
        ctx_texts.append(header + c.chunk_text)
    ctx_block = "\n\n".join(ctx_texts) if ctx_texts else "(Контекст відсутній)"
    return (
        "Нижче — релевантні уривки з документів. Використайте їх за можливості.\n\n"
        f"{ctx_block}\n\n"
        f"Запит користувача: {user_query}"
    )


# ------------------------- CLI / REPL -------------------------

def resolve_docs_path(args) -> Optional[str]:
    # Приоритет: CLI --docs > ENV DOCS_DIR
    docs = args.docs if args.docs else os.getenv("DOCS_DIR")
    if not docs:
        console.print("[red]Ошибка: укажите путь к документам через --docs или задайте DOCS_DIR в окружении.[/red]")
        return None
    if not os.path.isdir(docs):
        console.print(f"[red]Ошибка: папка не найдена: {docs}[/red]")
        return None
    return docs


def run_once(args, idx: Optional[RagIndex]):
    if idx is None or not idx.chunks:
        console.print("[red]Ошибка: индекс пуст или не построен.[/red]")
        sys.exit(2)
    contexts = idx.retrieve(args.text, top_k=args.top_k) if args.text else []
    if not contexts:
        console.print("[red]Ошибка: не найдено релевантных фрагментов в документах.[/red]")
        sys.exit(3)
    messages = [
        {"role": "system", "content": args.system or build_system_prompt()},
        {"role": "user", "content": build_user_prompt(args.text, contexts)},
    ]
    console.rule("Ответ")
    _ = llm_complete(model=args.model, messages=messages, temperature=args.temperature,
                     max_tokens=args.max_tokens, stream=True)


def run_repl(args, idx: Optional[RagIndex]):
    if idx is None or not idx.chunks:
        console.print("[red]Ошибка: индекс пуст или не построен.[/red]")
        sys.exit(2)
    console.print(Panel.fit(Text(f"DocChat REPL — модель: {args.model}", style="bold cyan")))
    history: List[dict] = [{"role": "system", "content": args.system or build_system_prompt()}]

    while True:
        try:
            q = Prompt.ask("[bold green]>[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]До зустрічі![/cyan]")
            break
        if not q.strip():
            continue
        if q.strip().lower() in {":q", ":quit", ":exit"}:
            break
        if q.strip().lower() == ":reload":
            console.print("[yellow]Перестроение индекса...[/yellow]")
            idx = build_index(args)  # переиспользует resolve_docs_path внутри
            if idx is None or not idx.chunks:
                console.print("[red]Ошибка: индекс пуст после перестроения.[/red]")
                continue
            else:
                console.print("[green]✔ Индекс обновлён.[/green]")
            continue

        contexts = idx.retrieve(q, top_k=args.top_k) if idx else []
        if not contexts:
            console.print("[red]Ошибка: не найдено релевантных фрагментов. Ответ не будет сформирован.[/red]")
            continue

        user_content = build_user_prompt(q, contexts)
        history.append({"role": "user", "content": user_content})

        console.rule("Ответ")
        answer = llm_complete(model=args.model, messages=history[-(2*args.history+1):],
                              temperature=args.temperature, max_tokens=args.max_tokens, stream=True)
        history.append({"role": "assistant", "content": answer})


def build_index(args) -> Optional[RagIndex]:
    docs_dir = resolve_docs_path(args)
    if not docs_dir:
        return None
    idx = RagIndex()
    idx.build(docs_dir, max_chars=args.chunk_chars, overlap=args.overlap)
    if not idx.chunks:
        return None
    return idx


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DocChat CLI — інтерактивна робота з документами через OpenRouter")
    p.add_argument("-m", "--model", required=True, help="ID модели OpenRouter (точный). Пример: openrouter/meta-llama/llama-3.1-8b-instruct")
    p.add_argument("-t", "--text", help="Одноразовый запрос и выход")
    p.add_argument("--interactive", action="store_true", help="REPL режим")
    # --docs теперь опционален: можно через DOCS_DIR
    p.add_argument("--docs", help="Каталог с документами (альтернатива: переменная окружения DOCS_DIR)")
    p.add_argument("--top_k", type=int, default=4, help="Сколько фрагментов брать в контекст")
    p.add_argument("--chunk_chars", type=int, default=2000, help="Длина чанка для индексации")
    p.add_argument("--overlap", type=int, default=200, help="Перекрытие между чанками")
    p.add_argument("--temperature", type=float, default=0.2, help="Temperature")
    p.add_argument("--max_tokens", type=int, default=800, help="Максимум токенов в ответе")
    p.add_argument("--history", type=int, default=6, help="Сколько последних пар сообщений держать в истории")
    p.add_argument("--system", help="Кастомная системная инструкция")
    return p.parse_args(argv)


def main(argv: List[str]):
    load_dotenv()  # .env поддержка
    args = parse_args(argv)

    # Если не указан -t и не указан --interactive, переходим в REPL
    if not args.text and not args.interactive:
        console.print("[yellow]Не указан -t и не выбран --interactive. Запускаю REPL.[/yellow]")
        args.interactive = True

    # Построение индекса (использует --docs или DOCS_DIR)
    idx = build_index(args)
    if idx:
        console.print("[green]Индекс готов[/green]")

    if args.text and not args.interactive:
        run_once(args, idx)
    else:
        run_repl(args, idx)


if __name__ == "__main__":
    main(sys.argv[1:])
