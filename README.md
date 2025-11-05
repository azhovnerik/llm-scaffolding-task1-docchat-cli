# DocChat CLI (RAG-STRICT via OpenRouter)
Quick start:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...
export DOCS_DIR=/path/to/docs
python docchat_cli.py -m openrouter/meta-llama/llama-3.1-8b-instruct -t "Hello"
