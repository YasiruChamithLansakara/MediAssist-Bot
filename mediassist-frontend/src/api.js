/* ==========================================================================
   API client
   --------------------------------------------------------------------------
   One place that knows how to talk to the backend, so the access code, the
   timeout and the error shape are handled identically everywhere.

   The backend returns structured errors as
       { error: { code, message, details }, request_id }
   which is far more useful in the UI than a bare status code — a 422 for an
   unsupported disease should not read the same as a 422 for a bad age.
   ========================================================================== */

const API_BASE = import.meta.env.VITE_API_BASE || "/api";
const ACCESS_STORAGE_KEY = "mediassist.access_code";
const DEFAULT_TIMEOUT = 30000;

/** Access code lives in localStorage so a reload does not re-prompt. */
export const accessCode = {
  get() {
    try {
      return localStorage.getItem(ACCESS_STORAGE_KEY) || "";
    } catch {
      return "";
    }
  },
  set(code) {
    try {
      if (code) localStorage.setItem(ACCESS_STORAGE_KEY, code);
      else localStorage.removeItem(ACCESS_STORAGE_KEY);
    } catch {
      /* private browsing — the code simply won't persist */
    }
  },
  clear() {
    this.set("");
  },
};

export class ApiError extends Error {
  constructor(message, { status = 0, code = "unknown", details = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }

  /** True when the caller should send the user back to the access gate. */
  get isAuthError() {
    return this.status === 401 || this.code === "access_denied";
  }
}

async function request(path, { method = "GET", body, timeout = DEFAULT_TIMEOUT, isForm } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  const headers = {};
  const code = accessCode.get();
  if (code) headers["X-Access-Code"] = code;
  if (body && !isForm) headers["Content-Type"] = "application/json";

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: isForm ? body : body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });

    let payload = null;
    const text = await response.text();
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = { raw: text };
      }
    }

    if (!response.ok) {
      const err = payload?.error || {};
      throw new ApiError(err.message || `Request failed (${response.status})`, {
        status: response.status,
        code: err.code || "http_error",
        details: err.details ?? payload,
      });
    }
    return payload;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error.name === "AbortError") {
      throw new ApiError(
        "The server took too long to respond. The first request after startup can be slow while models load — try again.",
        { code: "timeout" },
      );
    }
    throw new ApiError("Cannot reach the MediAssist server. Is the backend running?", {
      code: "network",
    });
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  /** Public — tells the UI whether to show the access gate at all. */
  config: () => request("/config", { timeout: 8000 }),
  health: () => request("/health", { timeout: 8000 }),
  meta: () => request("/meta", { timeout: 15000 }),
  dashboard: () => request("/dashboard", { timeout: 15000 }),

  lookup: (drug, disease, age) =>
    request(
      `/lookup?drug=${encodeURIComponent(drug)}&disease=${encodeURIComponent(
        disease,
      )}&age=${encodeURIComponent(age)}`,
      { timeout: 20000 },
    ),

  chat: (payload) => request("/chat", { method: "POST", body: payload, timeout: 60000 }),

  forgetChat: (sessionId) =>
    request("/chat/forget", { method: "POST", body: { session_id: sessionId } }),

  interactions: (drugs, disease) =>
    request("/interactions", { method: "POST", body: { drugs, disease }, timeout: 25000 }),

  prescriptionImage: (file, disease, age) => {
    const form = new FormData();
    form.append("file", file);
    form.append("disease", disease);
    form.append("age", String(age));
    return request("/prescription", { method: "POST", body: form, isForm: true, timeout: 120000 });
  },

  prescriptionText: (text, disease, age) =>
    request("/prescription/analyze-text", {
      method: "POST",
      body: { text, disease, age: Number(age) },
      timeout: 45000,
    }),
};

/** Verify a code by calling a gated endpoint with it. */
export async function verifyAccessCode(code) {
  accessCode.set(code);
  try {
    await api.meta();
    return true;
  } catch (error) {
    if (error instanceof ApiError && error.isAuthError) {
      accessCode.clear();
      return false;
    }
    throw error; // network or server problem — not a wrong code
  }
}
