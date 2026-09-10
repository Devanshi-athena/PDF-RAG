import { useEffect, useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
const NOT_FOUND_ANSWER = "I couldn't find that in the uploaded PDF.";

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body?.detail?.message || body?.detail || "Request failed");
  return body;
}

function App() {
  const [threads, setThreads] = useState([]);
  const [active, setActive] = useState(null);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState("");
  const [recording, setRecording] = useState(false);
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
      await request(`/threads/${threadId}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text }) });
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
  const speak = (text) => { if ("speechSynthesis" in window) { window.speechSynthesis.cancel(); window.speechSynthesis.speak(new SpeechSynthesisUtterance(text)); } };
  const removeThread = async (id) => { try { await request(`/threads/${id}`, { method: "DELETE" }); await refresh(); } catch (e) { setError(e.message); } };

  const visibleMessages = [...(active?.messages || []), ...(pendingByThread[active?.id] || [])];

  return <div className="app">
    <aside className="sidebar">
      <div className="brand"><strong>PDF RAG</strong></div>
      <button className="new-thread" onClick={newThread}>+ New thread</button>
      <div className="thread-list">{threads.map((thread) => <div className={`thread ${thread.id === active?.id ? "selected" : ""}`} key={thread.id}>
        <button onClick={() => setActive(thread)}><span>{thread.pdf_name ? "[doc]" : "[ ]"}</span>{thread.name}</button>
        <button className="delete" title="Delete thread" onClick={() => removeThread(thread.id)}>x</button>
      </div>)}</div>
      <small className="side-note">Documents and chat history stay local.</small>
    </aside>
    <main className="main">
      {!active ? <div className="empty"><h1>PDF RAG Assistant</h1><p>Create a thread and upload a PDF to get started.</p></div> :
      <>
        <header className="header"><div><h1>{active.name}</h1><p>{active.document ? `${active.document.name} - ${active.document.pages} pages` : "Upload a PDF to begin"}</p></div>
          {!active.document && (processing ? <div className="processing-status" role="status"><span className="spinner" />Processing...</div> : <label className="upload">Upload PDF<input type="file" accept="application/pdf" onChange={upload} disabled={busy || processing} /></label>)}
        </header>
        <section className="messages">
          {!visibleMessages.length && <div className="welcome"><div className="welcome-icon">[doc]</div><h2>Ask questions about your document</h2><p>Answers are grounded in the uploaded PDF and include page sources.</p></div>}
          {visibleMessages.map((message, index) => <article className={`message ${message.role}`} key={`${message.role}-${index}-${message.content}`}><div className="avatar">{message.role === "user" ? "You" : "AI"}</div><div className="bubble"><div className="content">{message.content}</div>
            {message.role === "assistant" && !message.pending && <><button className="listen" onClick={() => speak(message.content)}>Listen</button>{message.content.trim() !== NOT_FOUND_ANSWER && message.sources?.length > 0 && <details><summary>Sources ({message.sources.length})</summary>{message.sources.map((source, i) => <div className="source" key={i}><b>Page {source.page}</b><span>{source.text.slice(0, 300)}{source.text.length > 300 ? "..." : ""}</span></div>)}</details>}</>}
            {message.role === "assistant" && message.pending && <button className="listen loading">Loading…</button>}
          </div></article>)}
          <div ref={endRef} />
        </section>
        {error && <div className="error">{error}<button onClick={() => setError("")}>x</button></div>}
        <div className="composer"><button className={recording ? "mic recording" : "mic"} onClick={startRecording} disabled={!active.document || busy}>{recording ? "[stop]" : "[mic]"}</button><input value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === "Enter" && send()} placeholder={active.document ? "Ask a question about the PDF..." : "Upload a PDF first"} disabled={!active.document || busy} /><button className="send" onClick={() => send()} disabled={!active.document || busy || !question.trim()}>{busy ? "..." : "send"}</button></div>
      </>}
    </main>
  </div>;
}

export default App;
