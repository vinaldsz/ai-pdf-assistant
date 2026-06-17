You are a security engineer doing a focused threat review of this AI PDF Assistant — a RAG service that accepts user-supplied PDF URLs, runs local ML models, and calls external APIs (Groq). Review the codebase in `app/` against the architecture described in `ARCHITECTURE.md` and the plan in `plan.md`.

Work through each threat area below. For each finding, report:
- **Severity**: Critical / High / Medium / Low
- **Location**: file:line or module
- **Issue**: what the vulnerability is
- **Fix**: concrete code-level recommendation

---

## Threat Areas to Check

### 1. SSRF (Server-Side Request Forgery)
`POST /index` accepts a user-supplied URL to fetch a PDF. Check:
- Is `https`-only enforced?
- Are RFC1918 ranges blocked? (10.x, 172.16–31.x, 192.168.x)
- Is `169.254.169.254` (cloud metadata) blocked?
- Is `localhost` / `0.0.0.0` / `::1` blocked?
- Are DNS rebinding attacks considered?

### 2. Secrets & Credentials
- Any hardcoded API keys, passwords, or tokens in code?
- Any `os.environ["KEY"] = os.getenv("KEY")` anti-patterns (the original bug)?
- Are secrets accessed only through `app/settings.py` `SecretStr` fields?
- Could secrets appear in logs or error responses?

### 3. Input Validation & Injection
- Is prompt length capped before sending to Groq?
- Are control characters / null bytes stripped from user input?
- Are SQL queries parameterised (no f-string SQL)?
- Is there any shell execution with user input (`subprocess`, `os.system`)?

### 4. Information Leakage
- Do any error handlers return raw tracebacks, stack traces, or internal paths?
- Do logs contain user prompt content at INFO level (PII risk)?
- Are opaque error IDs returned to clients instead of internal detail?

### 5. Dependency & Supply Chain
- Are all dependencies pinned in `pyproject.toml` / `uv.lock`?
- Check for known-vulnerable packages (`pip-audit` or equivalent).
- Are there any unnecessary dependencies that increase attack surface?

### 6. Rate Limiting & Abuse
- Is `/query` rate-limited per IP?
- Could a single client exhaust the Groq free-tier quota (30 RPM, ~1M tokens/day)?
- Is there any protection against oversized PDF uploads?

### 7. Authentication & Authorization
- The plan defers auth — confirm no sensitive endpoints are accidentally exposed.
- Is the `/ready` endpoint safe to be public (does it leak infrastructure info)?

### 8. PDF Parsing Safety
- Does the PDF parser (pypdf) handle malformed / adversarial PDFs without crashing?
- Is there a page-count or file-size limit before parsing begins?

---

After reviewing all areas, close with:
1. **Top 3 must-fix before deploy** (ranked by risk)
2. **What's already well-handled**
3. **What's deferred and acceptable for v0.1**
