import { useEffect, useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
const NOT_FOUND_ANSWER = "I couldn't find that in the uploaded PDF.";

function Icon({ name, size = 16 }) {
  const paths = {
    file: <><path d="M5 2.75h6l3 3V15a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V3.75a1 1 0 0 1 1-1Z" /><path d="M11 2.75v3h3M6.5 9h5M6.5 12h5" /></>,
    mic: <><path d="M8 2.5a2 2 0 0 1 2 2v4a2 2 0 1 1-4 0v-4a2 2 0 0 1 2-2Z" /><path d="M3.75 8.5a4.25 4.25 0 0 0 8.5 0M8 13v2.5M5.75 15.5h4.5" /></>,
    play: <path d="m6 4 8 4-8 4V4Z" />,
    plus: <><path d="M8 3v10M3 8h10" /></>,
  };
  return <svg className={`icon icon-${name}`} width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}
const isTableLine = (line) => line.trim().startsWith("|");
const isSeparatorRow = (cells) => cells.length > 0 && cells.every((cell) => /^:?-{2,}:?$/.test(cell));
const splitRow = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());

// Answers are plain text; markdown tables (2+ lines starting with "|") are shown as real tables.
function MessageContent({ text }) {
  const blocks = [];
  let lines = [];
  let table = [];
  const flushText = () => { if (lines.length) blocks.push({ type: "text", value: lines.join("\n") }); lines = []; };
  const flushTable = () => {
    if (table.length >= 2) { flushText(); blocks.push({ type: "table", rows: table }); }
    else lines.push(...table);
    table = [];
  };
  for (const line of text.split("\n")) {
    if (isTableLine(line)) table.push(line);
    else { flushTable(); lines.push(line); }
  }
  flushTable();
  flushText();
  return blocks.map((block, index) => {
    if (block.type === "text") return block.value.trim() ? <div className="content" key={index}>{block.value.replace(/^\n+|\n+$/g, "")}</div> : null;
    const rows = block.rows.map(splitRow);
    const hasHeader = rows.length > 1 && isSeparatorRow(rows[1]);
    const header = hasHeader ? rows[0] : null;
    const body = rows.filter((cells, i) => !isSeparatorRow(cells) && !(hasHeader && i === 0));
    return <div className="table-wrap" key={index}><table className="answer-table">
      {header && <thead><tr>{header.map((cell, i) => <th key={i}>{cell}</th>)}</tr></thead>}
      <tbody>{body.map((cells, r) => <tr key={r}>{cells.map((cell, i) => <td key={i}>{cell}</td>)}</tr>)}</tbody>
    </table></div>;
  });
}

// Read tables aloud as sentences rather than pipe characters.
const speechText = (text) => text
  .split("\n")
  .filter((line) => !(isTableLine(line) && isSeparatorRow(splitRow(line))))
  .map((line) => (isTableLine(line) ? splitRow(line).filter(Boolean).join(", ") + "." : line))
  .join("\n");

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body?.detail?.message || body?.detail || "Request failed");
  return body;
}
async function streamChat(threadId, question, onToken, onDone) {
  const response = await fetch(`${API}/threads/${threadId}/chat/stream`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body?.detail?.message || body?.detail || "Request failed");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const events = buffer.split("\n\n"); buffer = events.pop() || "";
    for (const event of events) {
      const lines = event.split("\n");
      const type = lines.find((line) => line.startsWith("event:"))?.slice(6).trim();
      const data = lines.find((line) => line.startsWith("data:"))?.slice(5).trim();
      if (!data) continue;
      const payload = JSON.parse(data);
      if (type === "token") onToken(payload);
      if (type === "done") onDone(payload);
      if (type === "error") throw new Error(payload.message || "Chat generation failed");
    }
    if (done) break;
  }
}

function App() {
  const [threads, setThreads] = useState([]);
  const [active, setActive] = useState(null);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");
  const [recording, setRecording] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [pendingByThread, setPendingByThread] = useState({});
  const recorder = useRef(null);
  const chunks = useRef([]);
  const endRef = useRef(null);

  const refresh = async (selectId) => {
    const list = await request("/threads");
    setThreads(list);
    const selected = list.find((item) => item.id === (selectId || active?.id)) || list[0];
    setActive(selected || null);
  };
  useEffect(() => { refresh().catch((e) => setError(e.message)); }, []);
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [active?.messages?.length]);

  const newThread = async () => {
    try { setError(""); const thread = await request("/threads", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" }); await refresh(thread.id); }
    catch (e) { setError(e.message); }
  };
  const upload = async (event) => {
    const file = event.target.files[0]; if (!file || !active) return;
    const data = new FormData(); data.append("file", file);
    try {
      setProcessing(true); setError("");
      const updated = await request(`/threads/${active.id}/document`, { method: "POST", body: data });
      setActive(updated); await refresh(updated.id);
    } catch (e) { setError(e.message); } finally {
      setProcessing(false); event.target.value = "";
    }
  };

  const addPendingMessages = (threadId, text) => {
    setPendingByThread((current) => ({
      ...current,
      [threadId]: [
        ...(current[threadId] || []),
        { role: "user", content: text, pending: true },
        { role: "assistant", content: "....", pending: true }
      ]
    }));
  };

  const clearPendingMessages = (threadId) => {
    setPendingByThread((current) => {
      const next = { ...current };
      delete next[threadId];
      return next;
    });
  };

  const send = async (text = question) => {
    if (!active || !text.trim() || busy) return;
    const threadId = active.id;
    addPendingMessages(threadId, text);
    try {
      setBusy(true); setError(""); setQuestion("");
      let streamedAnswer = "";
      await streamChat(threadId, text, (token) => {
        streamedAnswer += token;
        setPendingByThread((current) => ({
          ...current,
          [threadId]: [
            ...(current[threadId] || []).slice(0, -1),
            { role: "assistant", content: streamedAnswer, pending: true },
          ],
        }));
      }, () => {});
      clearPendingMessages(threadId);
      await refresh(threadId);
    } catch (e) {
      clearPendingMessages(threadId);
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  const startRecording = async () => {
    if (recording) { recorder.current?.stop(); return; }
    if (!navigator.mediaDevices?.getUserMedia) { setError("Microphone recording is not supported by this browser."); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const media = new MediaRecorder(stream); chunks.current = [];
      media.ondataavailable = (event) => chunks.current.push(event.data);
      media.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop()); setRecording(false);
        const blob = new Blob(chunks.current, { type: media.mimeType || "audio/webm" });
        const data = new FormData(); data.append("file", blob, "recording.webm");
        try {
          const result = await request(`/threads/${active.id}/transcribe`, { method: "POST", body: data });
          setQuestion(result.text);
          await send(result.text);
        } catch (e) { setError(e.message); }
      };
      recorder.current = media; media.start(); setRecording(true);
    } catch (e) { setError("Microphone permission was denied."); }
  };
  const speak = (text) => {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(speechText(text));
    utterance.onstart = () => setSpeaking(true);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    window.speechSynthesis.speak(utterance);
  };
  const removeThread = async (id) => { try { await request(`/threads/${id}`, { method: "DELETE" }); await refresh(); } catch (e) { setError(e.message); } };

  const visibleMessages = [...(active?.messages || []), ...(pendingByThread[active?.id] || [])];

  return <div className="app">
    <aside className="sidebar">
      <div className="brand"><img src="/aplyd-logo.png" alt="APLYD by Athena Infonomics" /><div><strong>PDF RAG</strong><span>Document assistant</span></div></div>
      <button className="new-thread" onClick={newThread}><Icon name="plus" size={15} /> New thread</button>
      <div className="thread-list">{threads.map((thread) => <div className={`thread ${thread.id === active?.id ? "selected" : ""}`} key={thread.id}>
        <button onClick={() => setActive(thread)}><span className="thread-icon"><Icon name="file" size={15} /></span>{thread.name}</button>
        <button className="delete" title="Delete thread" aria-label={`Delete ${thread.name}`} onClick={() => removeThread(thread.id)}>×</button>
      </div>)}</div>
      <small className="side-note">Documents and chat history stay local.</small>
    </aside>
    <main className="main">
      {!active ? <div className="empty"><h1>PDF RAG Assistant</h1><p>Create a thread and upload a PDF to get started.</p></div> :
      <>
        <header className="header"><div className="document-heading"><div className="document-icon"><Icon name="file" size={21} /></div><div><h1>{active.name}</h1><p>{active.document ? <><span>{active.document.name}</span><span className="header-meta">{active.document.pages} {active.document.pages === 1 ? "page" : "pages"} · <b>Ready</b></span></> : "Upload a PDF to begin"}</p></div></div>
          {!active.document && (processing ? <div className="processing-status" role="status"><span className="spinner" />Processing...</div> : <label className="upload"><Icon name="file" size={15} /> Upload PDF<input type="file" accept="application/pdf" onChange={upload} disabled={busy || processing} /></label>)}
        </header>
        <section className="messages">
          {!visibleMessages.length && <div className="welcome"><div className="welcome-icon"><Icon name="file" size={22} /></div><h2>Ask questions about your document</h2><p>Answers are grounded in the uploaded PDF and include page sources.</p></div>}
          {visibleMessages.map((message, index) => <article className={`message ${message.role}`} key={`${message.role}-${index}-${message.content}`}><div className="avatar">{message.role === "user" ? "You" : "AI"}</div><div className="bubble"><MessageContent text={message.content} />
            {message.role === "assistant" && !message.pending && <><button className={speaking ? "listen active" : "listen"} onClick={() => speak(message.content)}><Icon name="play" size={12} /> {speaking ? "Playing" : "Listen"}</button>{message.content.trim() !== NOT_FOUND_ANSWER && message.sources?.length > 0 && <details><summary>Sources ({message.sources.length})</summary>{message.sources.map((source, i) => <div className="source" key={i}><span className="source-page">Page {source.page}</span><span>{source.text.slice(0, 300)}{source.text.length > 300 ? "..." : ""}</span></div>)}</details>}</>}
            {message.role === "assistant" && message.pending && <button className="listen loading">Loading…</button>}
          </div></article>)}
          <div ref={endRef} />
        </section>
        {error && <div className="error">{error}<button onClick={() => setError("")}>x</button></div>}
        <div className="composer"><button className={recording ? "mic recording" : "mic"} onClick={startRecording} disabled={!active.document || busy} aria-label={recording ? "Stop recording" : "Start voice question"}><Icon name="mic" size={18} /><span className="control-label">{recording ? "Stop" : "Voice"}</span></button><input value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()} placeholder={active.document ? "Ask a question about the PDF..." : "Upload a PDF first"} disabled={!active.document || busy} /><button className="send" onClick={() => send()} disabled={!active.document || busy || !question.trim()}>{busy ? "Sending..." : "Send"}</button></div>
      </>}
    </main>
  </div>;
}

export default App;
