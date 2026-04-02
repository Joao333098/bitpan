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
    active_sessions[session_id] = {
        "agent": None,
        "running": False,
        "human_input_queue": human_input_queue,
    }

    async def send(msg: dict):
        try:
            await websocket.send_text(json.dumps(msg))
        except Exception:
            pass

    def get_screenshot():
        sess = active_sessions.get(session_id, {})
        agent = sess.get("agent")
        if agent and agent.browser._state and agent.browser._state.screenshot_with_highlights:
            return agent.browser._state.screenshot_with_highlights
        if agent and agent.browser._state and agent.browser._state.screenshot:
            return agent.browser._state.screenshot
        return None

    def get_url():
        sess = active_sessions.get(session_id, {})
        agent = sess.get("agent")
        if agent and agent.browser.current_page:
            try:
                return agent.browser.current_page.url
            except Exception:
                pass
        return ""

    async def stream_screenshots():
        """Background task: send live screenshots every 500ms while agent is running"""
        while True:
            await asyncio.sleep(0.5)
            sess = active_sessions.get(session_id, {})
            if not sess.get("running"):
                break
            ss = get_screenshot()
            if ss:
                try:
                    await websocket.send_text(json.dumps({"type": "live_screenshot", "screenshot": ss, "url": get_url()}))
                except Exception:
                    break

    async def run_agent(prompt: str):
        agent = None
        stream_task = None
        try:
            llm = make_nova_provider()
            browser_config = BrowserConfig(viewport_size={"width": 1280, "height": 800})
            agent = Agent(llm=llm, browser_config=browser_config)

            if session_id in active_sessions:
                active_sessions[session_id]["agent"] = agent

            await send({"type": "status", "content": "running"})

            stream_task = asyncio.create_task(stream_screenshots())

            step = 0
            result = None
            is_done = False
            max_steps = 50

            await agent._setup_messages(prompt)

            while not is_done and step < max_steps:
                await send({"type": "step_start", "step": step + 1})

                # Retry failed steps up to 3 times
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
                    await send({"type": "error", "content": f"Step {step+1} failed after 3 attempts: {step_error}"})
                    break

                step += 1

                screenshot_b64 = get_screenshot()
                current_url = get_url()

                # ── HUMAN CONTROL / CAPTCHA DETECTED ──────────────────────────
                if result.give_control:
                    agent_message = str(result.content or "The agent needs your help to continue.")
                    await send({
                        "type": "human_control",
                        "step": step,
                        "message": agent_message,
                        "screenshot": screenshot_b64,
                        "url": current_url,
                    })

                    # Wait up to 3 min for the user to reply
                    try:
                        user_text = await asyncio.wait_for(human_input_queue.get(), timeout=180)
                    except asyncio.TimeoutError:
                        await send({"type": "error", "content": "Timeout: no response from user in 3 minutes."})
                        break

                    # Type the user's input into the page
                    page = agent.browser.current_page
                    if page and user_text:
                        try:
                            typed = False
                            # Priority order: captcha-specific → generic text → password → textarea
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
                                                await el.fill("")          # clear first
                                                await el.type(user_text, delay=50)
                                                await asyncio.sleep(0.3)
                                                await page.keyboard.press("Enter")
                                                typed = True
                                                logger.info(f"Typed into element: {selector}")
                                                break
                                        except Exception:
                                            continue
                                    if typed:
                                        break
                            if not typed:
                                # Last resort: type into whatever is currently focused
                                logger.info("No visible input found, typing into focused element")
                                await page.keyboard.type(user_text, delay=50)
                                await page.keyboard.press("Enter")
                            await asyncio.sleep(2)
                        except Exception as e:
                            logger.warning(f"Could not fill human input: {e}")

                    # Reset so the loop continues
                    result = ActionResult(is_done=False, content=f"User provided input. Continuing.")
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

                # ── NORMAL STEP ───────────────────────────────────────────────
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
            if stream_task and not stream_task.done():
                stream_task.cancel()
            if agent:
                try:
                    await agent.browser.close()
                except Exception:
                    pass
            if session_id in active_sessions:
                active_sessions[session_id]["running"] = False
                active_sessions[session_id]["agent"] = None
            await send({"type": "status", "content": "idle"})

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

            elif msg.get("type") == "stop":
                agent = active_sessions.get(session_id, {}).get("agent")
                if agent:
                    try:
                        await agent.browser.close()
                    except Exception:
                        pass
                if session_id in active_sessions:
                    active_sessions[session_id]["running"] = False
                    active_sessions[session_id]["agent"] = None
                await send({"type": "status", "content": "idle"})
                await send({"type": "error", "content": "Stopped by user."})

    except WebSocketDisconnect:
        pass
    finally:
        agent = active_sessions.pop(session_id, {}).get("agent")
        if agent:
            try:
                await agent.browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
