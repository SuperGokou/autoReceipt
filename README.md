<h1 align="center">autoReceipt</h1>

<p align="center">
  <strong>Autonomous receipt-to-coupon pipeline powered by multi-agent AI and browser automation.</strong><br/>
  Upload a receipt photo, and the system extracts the survey URL, completes the feedback form with a vision LLM, and delivers the coupon code to your inbox.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-3-green" alt="Flask" />
  <img src="https://img.shields.io/badge/Playwright-Chromium-orange" alt="Playwright" />
  <img src="https://img.shields.io/badge/Qwen3--VL-Vision_LLM-purple" alt="Qwen3-VL" />
  <img src="https://img.shields.io/badge/Docker-Render-cyan" alt="Docker" />
</p>

<p align="center">
  <a href="https://autoreceipt-ql5e.onrender.com"><strong>Live Demo</strong></a>
</p>

---

## Overview

autoReceipt is a full-stack automation system that converts grocery and fast-food receipt photos into redeemable coupon codes. It combines computer vision, large language models, and headless browser automation in a coordinated multi-agent pipeline -- no manual survey completion required.

Upload a receipt image. The system extracts the survey URL, navigates and completes the feedback form autonomously using a vision LLM, and delivers the resulting coupon code to your inbox.

## Problem Statement

Most retail and QSR (quick-service restaurant) chains print a survey invitation on every receipt, offering a discount coupon in exchange for customer feedback. In practice, the conversion rate on these surveys is extremely low -- the process is time-consuming and repetitive, leaving significant value unclaimed.

autoReceipt automates the entire workflow end-to-end, turning a photo of any qualifying receipt into a usable coupon with zero manual intervention.

## Technical Highlights

- **Multi-modal ingestion** -- Layered extraction pipeline (QR detection, Tesseract OCR, vision LLM fallback) handles real-world receipt photos: crumpled, blurred, rotated, or partially obscured.
- **Vision-driven navigation** -- Instead of brittle CSS selectors, the Navigator agent uses Qwen3-VL to interpret each survey page visually and determine the correct form interactions, making it generalizable across unseen survey providers.
- **Multi-agent orchestration** -- Four specialized agents (Ingestion, Supervisor, Navigator, Fulfillment) operate as a pipeline with clear contracts, enabling independent testing and modular replacement.
- **Real-time observability** -- Server-Sent Events (SSE) stream pipeline progress, live terminal output, and browser preview frames to the web dashboard.
- **Adaptive backend selection** -- Automatic detection and fallback between local Ollama inference and DashScope cloud API (Alibaba Cloud Qwen-VL), with model-level retry logic for quota exhaustion.

## System Architecture

```
Receipt Image
      |
      v
+-------------+     +--------------+     +-------------+     +--------------+
|  Ingestion  | --> |  Supervisor  | --> |  Navigator  | --> | Fulfillment  |
|  Agent      |     |  Agent       |     |  Agent      |     |  Agent       |
+-------------+     +--------------+     +-------------+     +--------------+
| QR / OCR /  |     | Mood ->      |     | Vision LLM  |     | Code extract |
| Vision LLM  |     | Persona      |     | + Playwright |     | + Email      |
| URL extract  |     | mapping      |     | form fill    |     | delivery     |
+-------------+     +--------------+     +-------------+     +--------------+
```

| Stage | Agent | Responsibility |
|-------|-------|----------------|
| 1 | **Ingestion** | Extract survey URL from receipt image via QR code detection, OCR, or vision LLM fallback |
| 2 | **Supervisor** | Map the selected mood (Happy / Neutral / Angry) to a persona configuration with rating preferences and response tone |
| 3 | **Navigator** | Launch a headless Chromium instance, analyze each page with the vision model, fill forms, and advance through the survey |
| 4 | **Fulfillment** | Extract the coupon/validation code from the completion page and deliver it via email |

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | Vanilla JS, SSE streaming, two-panel dashboard |
| **Backend** | Python 3.10+, Flask, async pipeline orchestration |
| **AI / ML** | Qwen3-VL (local via Ollama or cloud via DashScope), Tesseract OCR, pyzbar QR |
| **Browser Automation** | Playwright (Chromium, headless) |
| **Email Delivery** | Formspree (default) or SMTP (Gmail, Outlook, Yahoo) |
| **CI/CD** | GitHub Actions (lint + test), Render deploy via webhook |
| **Containerization** | Docker, Render Blueprint (`render.yaml`) |

## Screenshots

**Dashboard -- Idle State**

![Web UI](ScreenShot.png)

**Automation Complete -- Coupon Extracted**

![Success](screenshots/success.png)

## Getting Started

### Prerequisites

- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Ollama](https://ollama.ai) with `qwen3-vl` model (optional -- falls back to DashScope cloud)

### Installation

```bash
pip install -r requirements.txt
playwright install
```

### Configuration

Create a `.env` file in the project root:

```env
# Email delivery (Formspree -- free tier, no SMTP server needed)
FORMSPREE_ENDPOINT=https://formspree.io/f/your-form-id

# OCR engine path
TESSERACT_CMD=/usr/bin/tesseract

# Vision model
OLLAMA_VISION_MODEL=qwen3-vl
# OLLAMA_REMOTE_HOST=http://your-server:11434   # remote Ollama, if applicable

# Browser
BROWSER_HEADLESS=true
```

See [FORMSPREE_SETUP.md](FORMSPREE_SETUP.md) for detailed email configuration.

### Usage

**Web UI** (recommended):

```bash
python run_web.py          # http://localhost:5000
python run_web.py --debug  # with hot-reload
```

**CLI:**

```bash
# From receipt image
python -m survey_bot run -i receipt.jpg -m happy -e user@email.com

# Direct URL (skip ingestion)
python -m survey_bot direct -u https://survey.example.com/abc -m happy -e user@email.com
```

**Windows:**

```bash
run_web.bat
```

## Testing

```bash
pytest tests/ -v                                 # unit tests
pytest tests/ -m integration -v                  # integration tests
RUN_E2E_TESTS=true pytest tests/test_e2e.py -v   # end-to-end
```

## Project Structure

```
src/survey_bot/
  agents/       # Ingestion, Supervisor, Fulfillment agents + Formspree sender
  browser/      # Playwright launcher, page interactor, observer
  llm/          # LLM chains, navigation graph, vision navigator, prompts
  models/       # Pydantic models (receipt, persona, page state, survey result)
  utils/        # QR decoder, image processing
  web/          # Flask app and HTML templates
  main.py       # SurveyBot orchestrator
```

## Deployment

### CI/CD

GitHub Actions workflows are included:

- **CI** (`.github/workflows/ci.yml`) -- Runs `mypy` and `pytest` on every push to `main` and on all pull requests. Tests marked `requires_llm` or `integration` are skipped in CI.
- **Deploy** (`.github/workflows/deploy.yml`) -- Triggers a Render deploy via webhook on push to `main`.

### Render (Production)

The app is containerized via `Dockerfile` and configured with `render.yaml` (Render Blueprint).

**Required environment variables (Render dashboard):**

| Variable | Description |
|----------|-------------|
| `DASHSCOPE_API_KEY` | API key from [DashScope](https://dashscope.console.aliyun.com/) for Qwen-VL cloud inference |
| `FORMSPREE_ENDPOINT` | Formspree form endpoint for email delivery |

**Required GitHub secret:**

| Secret | Description |
|--------|-------------|
| `RENDER_DEPLOY_HOOK_URL` | Deploy hook URL from Render dashboard (Settings > Deploy Hook) |

### Vision Backend Auto-Detection

The system automatically selects the optimal vision inference backend:

1. **Explicit override** -- `VISION_BACKEND` env var set to `ollama` or `dashscope`
2. **Local Ollama** -- `localhost:11434` is reachable
3. **DashScope cloud** -- `DASHSCOPE_API_KEY` is set (Alibaba Cloud Qwen-VL)
4. **Remote Ollama** -- `OLLAMA_REMOTE_HOST` is set

For local development, running Ollama is sufficient -- no additional configuration needed. For cloud deployment, set `DASHSCOPE_API_KEY` to use managed Qwen-VL inference without provisioning GPU infrastructure.

### Email Delivery

Default: [Formspree](https://formspree.io/) (free tier). No SMTP server or credentials required. Coupon codes are delivered through the configured Formspree form endpoint. SMTP (Gmail, Outlook, Yahoo) is also supported as an alternative.

## License

This project is for personal and educational use.
