import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { Alert, Markdown, Pill } from "./ui.jsx";
import { Icon } from "./icons.jsx";

const STARTERS = [
  "What are the side effects of metformin?",
  "Is amlodipine safe for elderly patients?",
  "Can I take ibuprofen with warfarin?",
  "What does BD mean on my prescription?",
];

export default function Chat({ disease, age, ready }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const streamRef = useRef(null);

  // Keep the newest message in view, but never yank the page while the user
  // is reading something further up.
  useEffect(() => {
    const el = streamRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 220;
    if (nearBottom || busy) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  async function send(text) {
    const message = (text ?? input).trim();
    if (!message || !ready || busy) return;

    setMessages((prev) => [...prev, { role: "user", text: message }]);
    setInput("");
    setBusy(true);
    setError("");

    try {
      const data = await api.chat({
        disease,
        age: Number(age),
        message,
        session_id: sessionId || undefined,
      });
      if (data.session_id) setSessionId(data.session_id);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.answer,
          emergency: data.is_emergency,
          source: data.answer_source,
          intent: data.intent,
          drugs: (data.matched_drugs || [])
            .map((m) => m.best_match?.generic_name_clean)
            .filter(Boolean),
          unconfirmed: (data.matched_drugs || []).some((m) => m.needs_confirmation),
        },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function clearChat() {
    if (sessionId) {
      try {
        await api.forgetChat(sessionId);
      } catch {
        /* the session expires on its own regardless */
      }
    }
    setMessages([]);
    setSessionId(null);
    setError("");
  }

  return (
    <div className="stack">
      <div className="chat">
        <div className="card-head" style={{ borderRadius: 0 }}>
          <div>
            <h2>Ask about your medicines</h2>
            <p>
              Answers come from official drug label data — not from the model&rsquo;s memory.
            </p>
          </div>
          <div className="spacer" />
          {messages.length > 0 && (
            <button className="btn btn-quiet btn-sm" onClick={clearChat}>
              <Icon.Trash size={15} /> Clear
            </button>
          )}
        </div>

        <div className="chat-stream" ref={streamRef}>
          {messages.length === 0 && (
            <div className="empty" style={{ margin: "auto 0" }}>
              <Icon.Chat size={34} weight={1.4} />
              <h3>What would you like to know?</h3>
              <p>
                Ask about a medicine&rsquo;s purpose, dosage, side effects, warnings or interactions.
                This assistant only answers medication questions.
              </p>
            </div>
          )}

          {messages.map((msg, index) => (
            <div
              key={index}
              className={`msg ${msg.role === "user" ? "msg-user" : "msg-bot"} fade-in`}
            >
              <div className="msg-avatar">
                {msg.role === "user" ? "You" : <Icon.Pill size={15} />}
              </div>
              <div className="msg-body">
                {msg.emergency && (
                  <div className="emergency" style={{ marginBottom: 12 }}>
                    <h3>
                      <Icon.Alert size={19} /> Urgent — seek help now
                    </h3>
                    <div className="body">
                      <Markdown text={msg.text} />
                    </div>
                  </div>
                )}
                {!msg.emergency && <Markdown text={msg.text} />}

                {msg.role === "assistant" && !msg.emergency && (
                  <div className="msg-meta">
                    {msg.drugs?.length > 0 &&
                      msg.drugs.map((d) => (
                        <Pill key={d} tone="ok">
                          {d}
                        </Pill>
                      ))}
                    {msg.unconfirmed && <Pill tone="warn">unconfirmed match</Pill>}
                    <span className="tiny">
                      {msg.source === "llm_grounded" ? "grounded in label data" : "rule-based"}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}

          {busy && (
            <div className="msg msg-bot">
              <div className="msg-avatar">
                <Icon.Pill size={15} />
              </div>
              <div className="msg-body">
                <span className="typing">
                  <i />
                  <i />
                  <i />
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="chat-compose">
          {error && (
            <Alert tone="danger" title="Could not send">
              {error}
            </Alert>
          )}

          {messages.length === 0 && (
            <div className="suggestions">
              {STARTERS.map((s) => (
                <button key={s} className="suggestion" onClick={() => send(s)} disabled={!ready}>
                  {s}
                </button>
              ))}
            </div>
          )}

          <form
            className="compose-row"
            onSubmit={(e) => {
              e.preventDefault();
              send();
            }}
          >
            <input
              className="input"
              placeholder={ready ? "Ask about a medicine…" : "Choose a condition and age first"}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={!ready || busy}
            />
            <button className="btn btn-primary" disabled={!ready || busy || !input.trim()}>
              {busy ? <Icon.Spinner size={16} /> : <Icon.Send size={16} />}
              Send
            </button>
          </form>

          <div className="compose-hint">
            <span>Educational information only — always confirm with your pharmacist.</span>
          </div>
        </div>
      </div>
    </div>
  );
}
