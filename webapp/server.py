import asyncio
import json
import logging
import os
import sys

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from openai import AsyncOpenAI

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from index.agent.agent import Agent
from index.agent.models import ActionResult
from index.browser.browser import BrowserConfig
from index.llm.providers.openai import OpenAIProvider

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("webapp")

app = FastAPI()

NOVA_API_KEY = os.environ.get("NOVA_API_KEY", "")
NOVA_BASE_URL = "https://api.nova.amazon.com/v1"
NOVA_MODEL = "nova-2-lite-v1"


def make_nova_provider() -> OpenAIProvider:
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.model = NOVA_MODEL
    provider.reasoning_effort = None
    provider.client = AsyncOpenAI(api_key=NOVA_API_KEY, base_url=NOVA_BASE_URL)
    return provider


active_sessions: dict = {}


@app.get("/")
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(f.read())


@app.get("/favicon.ico")
async def favicon():
    return HTMLResponse("", status_code=204)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = id(websocket)
    human_input_queue: asyncio.Queue = asyncio.Queue()
    continue_event: asyncio.Event = asyncio.Event()
    active_sessions[session_id] = {
        "agent": None,
        "running": False,
        "paused": False,
        "human_input_queue": human_input_queue,
        "continue_event": continue_event,
    }

    async def send(msg: dict):
        try:
            await websocket.send_text(json.dumps(msg))
        except Exception:
            pass

    def get_agent():
        return active_sessions.get(session_id, {}).get("agent")

    def get_page():
        agent = get_agent()
        if agent and agent.browser.current_page:
            return agent.browser.current_page
        return None

    def get_url():
        page = get_page()
        if page:
            try:
                return page.url
            except Exception:
                pass
        return ""

    def get_screenshot():
        agent = get_agent()
        if agent and agent.browser._state:
            return agent.browser._state.screenshot_with_highlights or agent.browser._state.screenshot
        return None

    # ── CDP Screencast ──────────────────────────────────────────────
    async def start_screencast(agent: Agent):
        async def on_frame(jpeg_b64: str):
            try:
                await websocket.send_text(json.dumps({
                    "type": "live_frame",
                    "data": jpeg_b64,
                    "url": get_url(),
                }))
            except Exception:
                pass
        try:
            await agent.browser.start_screencast(on_frame)
        except Exception as e:
            logger.warning(f"Screencast start failed: {e}")

    # ── Agent runner ────────────────────────────────────────────────
    async def run_agent(prompt: str):
        agent = None
        try:
            llm = make_nova_provider()
            browser_config = BrowserConfig(viewport_size={"width": 1280, "height": 800})
            agent = Agent(llm=llm, browser_config=browser_config)

            if session_id in active_sessions:
                active_sessions[session_id]["agent"] = agent

            await send({"type": "status", "content": "running"})

            # Start live screencast
            await start_screencast(agent)

            step = 0
            result = None
            is_done = False
            max_steps = 50

            await agent._setup_messages(prompt)

            while not is_done and step < max_steps:
                await send({"type": "step_start", "step": step + 1})

                step_error = None
                for attempt in range(3):
                    try:
                        result, summary = await agent.step(step, result)
                        step_error = None
                        break
                    except Exception as e:
                        step_error = e
                        logger.warning(f"Step {step+1} attempt {attempt+1} failed: {e}")
                        await asyncio.sleep(1)

                if step_error:
                    await send({"type": "error", "content": f"Step {step+1} falhou: {step_error}"})
                    await send({"type": "status", "content": "paused"})
                    if session_id in active_sessions:
                        active_sessions[session_id]["paused"] = True
                        active_sessions[session_id]["running"] = False
                    ce = active_sessions.get(session_id, {}).get("continue_event")
                    if ce:
                        ce.clear()
                        try:
                            await asyncio.wait_for(ce.wait(), timeout=300)
                        except asyncio.TimeoutError:
                            break
                    if session_id in active_sessions:
                        active_sessions[session_id]["paused"] = False
                        active_sessions[session_id]["running"] = True
                    result = None
                    await send({"type": "status", "content": "running"})

                step += 1
                screenshot_b64 = get_screenshot()
                current_url = get_url()

                if result.give_control:
                    agent_message = str(result.content or "The agent needs your help to continue.")
                    await send({
                        "type": "human_control",
                        "step": step,
                        "message": agent_message,
                        "screenshot": screenshot_b64,
                        "url": current_url,
                    })
                    try:
                        user_text = await asyncio.wait_for(human_input_queue.get(), timeout=180)
                    except asyncio.TimeoutError:
                        await send({"type": "error", "content": "Timeout: no response from user in 3 minutes."})
                        break

                    page = get_page()
                    if page and user_text:
                        try:
                            typed = False
                            selector_groups = [
                                ['input[name*="captcha"]', 'input[id*="captcha"]',
                                 'input[placeholder*="captcha"]', 'input[class*="captcha"]'],
                                ['input[type="text"]', 'input:not([type])', 'input[type="search"]'],
                                ['input[type="password"]'],
                                ['textarea'],
                            ]
                            for group in selector_groups:
                                if typed:
                                    break
                                for selector in group:
                                    elements = await page.query_selector_all(selector)
                                    for el in elements:
                                        try:
                                            if await el.is_visible():
                                                await el.scroll_into_view_if_needed()
                                                await el.click()
                                                await el.fill("")
                                                await el.type(user_text, delay=50)
                                                await asyncio.sleep(0.3)
                                                await page.keyboard.press("Enter")
                                                typed = True
                                                break
                                        except Exception:
                                            continue
                                    if typed:
                                        break
                            if not typed:
                                await page.keyboard.type(user_text, delay=50)
                                await page.keyboard.press("Enter")
                            await asyncio.sleep(2)
                        except Exception as e:
                            logger.warning(f"Could not fill human input: {e}")

                    result = ActionResult(is_done=False, content="User provided input. Continuing.")
                    is_done = False
                    await send({
                        "type": "step",
                        "step": step,
                        "summary": f"Human helped: {user_text[:60]}",
                        "is_done": False,
                        "screenshot": get_screenshot(),
                        "url": get_url(),
                        "error": "",
                    })
                    continue

                is_done = result.is_done
                await send({
                    "type": "step",
                    "step": step,
                    "summary": summary or "",
                    "is_done": is_done,
                    "screenshot": screenshot_b64,
                    "url": current_url,
                    "error": result.error or "",
                })

                if is_done:
                    final_content = result.content
                    if isinstance(final_content, dict):
                        final_content = json.dumps(final_content, indent=2)
                    await send({
                        "type": "final",
                        "content": str(final_content) if final_content else "Task completed.",
                        "steps": step,
                    })
                    break

            if not is_done and step >= max_steps:
                await send({"type": "error", "content": f"Reached maximum {max_steps} steps without completing."})

        except Exception as e:
            logger.exception("Agent error")
            await send({"type": "error", "content": str(e)})
        finally:
            if agent:
                try:
                    await agent.browser.stop_screencast()
                except Exception:
                    pass
                try:
                    await agent.browser.close()
                except Exception:
                    pass
            if session_id in active_sessions:
                active_sessions[session_id]["running"] = False
                active_sessions[session_id]["agent"] = None
            await send({"type": "status", "content": "idle"})

    # ── WebSocket message loop ──────────────────────────────────────
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "run":
                if active_sessions.get(session_id, {}).get("running"):
                    await send({"type": "error", "content": "Agent is already running."})
                    continue
                prompt = msg.get("prompt", "").strip()
                if not prompt:
                    await send({"type": "error", "content": "Please enter a task."})
                    continue
                if session_id in active_sessions:
                    active_sessions[session_id]["running"] = True
                await send({"type": "status", "content": "starting"})
                asyncio.create_task(run_agent(prompt))

            elif msg.get("type") == "human_input":
                human_input_queue.put_nowait(msg.get("text", ""))

            # ── Mouse / keyboard / scroll events ──
            elif msg.get("type") == "mousedown":
                page = get_page()
                if page:
                    btn = {0: "left", 1: "middle", 2: "right"}.get(msg.get("button", 0), "left")
                    try:
                        await page.mouse.move(msg["x"], msg["y"])
                        await page.mouse.down(button=btn)
                    except Exception as e:
                        logger.debug(f"mousedown failed: {e}")

            elif msg.get("type") == "mouseup":
                page = get_page()
                if page:
                    btn = {0: "left", 1: "middle", 2: "right"}.get(msg.get("button", 0), "left")
                    try:
                        await page.mouse.up(button=btn)
                    except Exception as e:
                        logger.debug(f"mouseup failed: {e}")

            elif msg.get("type") == "click":
                page = get_page()
                if page:
                    btn = {0: "left", 1: "middle", 2: "right"}.get(msg.get("button", 0), "left")
                    try:
                        await page.mouse.click(msg["x"], msg["y"], button=btn)
                    except Exception as e:
                        logger.debug(f"click failed: {e}")

            elif msg.get("type") == "mousemove":
                page = get_page()
                if page:
                    try:
                        await page.mouse.move(msg["x"], msg["y"])
                    except Exception as e:
                        logger.debug(f"mousemove failed: {e}")

            elif msg.get("type") == "wheel":
                page = get_page()
                if page:
                    try:
                        await page.mouse.wheel(msg.get("deltaX", 0), msg.get("deltaY", 0))
                    except Exception as e:
                        logger.debug(f"wheel failed: {e}")

            elif msg.get("type") == "keydown":
                page = get_page()
                if page:
                    key = msg.get("key", "")
                    try:
                        mods = msg.get("modifiers", {})
                        combo = ""
                        if mods.get("ctrl"):  combo += "Control+"
                        if mods.get("alt"):   combo += "Alt+"
                        if mods.get("shift"): combo += "Shift+"
                        if mods.get("meta"):  combo += "Meta+"

                        if len(key) == 1:
                            await page.keyboard.type(key)
                        else:
                            special = {
                                "Enter": "Enter", "Backspace": "Backspace", "Tab": "Tab",
                                "Escape": "Escape", "ArrowLeft": "ArrowLeft", "ArrowRight": "ArrowRight",
                                "ArrowUp": "ArrowUp", "ArrowDown": "ArrowDown",
                                "Delete": "Delete", "Home": "Home", "End": "End",
                                "PageUp": "PageUp", "PageDown": "PageDown", " ": "Space",
                            }
                            mapped = special.get(key, key)
                            await page.keyboard.press(combo + mapped)
                    except Exception as e:
                        logger.debug(f"keydown failed: {e}")

            elif msg.get("type") == "navigate":
                page = get_page()
                if page:
                    action = msg.get("action")
                    try:
                        if action == "back":
                            await page.go_back()
                        elif action == "forward":
                            await page.go_forward()
                        elif action == "refresh":
                            await page.reload()
                        elif action == "goto":
                            url = msg.get("url", "")
                            if url and not url.startswith("http"):
                                url = "https://" + url
                            if url:
                                await page.goto(url, wait_until="domcontentloaded")
                    except Exception as e:
                        logger.debug(f"navigate failed: {e}")

            elif msg.get("type") == "force_continue":
                sess = active_sessions.get(session_id, {})
                ce = sess.get("continue_event")
                if ce and sess.get("paused"):
                    ce.set()

            elif msg.get("type") == "stop":
                ce = active_sessions.get(session_id, {}).get("continue_event")
                if ce:
                    ce.set()
                agent = active_sessions.get(session_id, {}).get("agent")
                if agent:
                    try:
                        await agent.browser.stop_screencast()
                    except Exception:
                        pass
                    try:
                        await agent.browser.close()
                    except Exception:
                        pass
                if session_id in active_sessions:
                    active_sessions[session_id]["running"] = False
                    active_sessions[session_id]["paused"] = False
                    active_sessions[session_id]["agent"] = None
                await send({"type": "status", "content": "idle"})
                await send({"type": "error", "content": "Stopped by user."})

    except WebSocketDisconnect:
        pass
    finally:
        agent = active_sessions.pop(session_id, {}).get("agent")
        if agent:
            try:
                await agent.browser.stop_screencast()
            except Exception:
                pass
            try:
                await agent.browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
