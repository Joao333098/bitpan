# Bitpan

Browser AI agent powered by Amazon Nova API. Features:
- Live browser screenshot view
- Real-time step-by-step activity log
- CAPTCHA handling: agent pauses and asks user to type the solution
- Console execution: agent reads and writes JS to the browser console
- Web search: Nova built-in grounding

## Deploy on Render

1. Connect this repo in your Render dashboard
2. Set the `NOVA_API_KEY` environment variable
3. Render will use the `render.yaml` config automatically

## Local Development

```bash
pip install -e .
playwright install chromium --with-deps
python webapp/server.py
```
