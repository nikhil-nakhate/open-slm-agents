"use client";
import { useState } from "react";

export default function Home() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<{role:"user"|"assistant", content:string}[]>([]);
  const [sources, setSources] = useState<any[]>([]);
  const [busy, setBusy] = useState(false);

  async function send() {
    if (!input.trim()) return;
    setMessages(m => [...m, { role:"user", content: input }]);
    setBusy(true);

    const res = await fetch("/api/chat", {
      method:"POST",
      headers: { "Content-Type":"application/json" },
      body: JSON.stringify({ message: input })
    });
    const data = await res.json();

    setMessages(m => [...m, { role:"assistant", content: data.answer }]);
    setSources(data.sources || []);
    setInput("");
    setBusy(false);
  }

  return (
    <main className="p-6 max-w-3xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Nutrition RAG Chat</h1>

      <div className="space-y-3 mb-4">
        {messages.map((m,i)=>(
          <div key={i} className={m.role==="user"?"text-right":"text-left"}>
            <div className={`inline-block px-4 py-2 rounded-lg ${
              m.role==="user" ? "bg-blue-600 text-white":"bg-gray-200"
            }`}>
              {m.content}
            </div>
          </div>
        ))}
      </div>

      <div className="flex gap-2">
        <input
          className="flex-1 border rounded px-3 py-2"
          value={input}
          onChange={e=>setInput(e.target.value)}
          onKeyDown={e=>e.key==="Enter" && send()}
          placeholder="Ask about the nutrition PDF..."
          disabled={busy}
        />
        <button onClick={send} disabled={busy} className="bg-blue-600 text-white px-4 rounded">
          Send
        </button>
      </div>

      {sources.length > 0 && (
        <div className="mt-6">
          <h2 className="font-semibold mb-2">Sources</h2>
          <ul className="space-y-2">
            {sources.map((s, i)=>(
              <li key={s.id} className="border rounded p-2 text-sm bg-gray-50">
  <strong>[{i+1}]</strong>
  {" "}Page: {s.metadata?.page ?? "?"}
  {" "}· sim: {typeof s.similarity === "number" ? s.similarity.toFixed(3) : "—"}
  {" "}— {s.content.slice(0, 150)}...
</li>

            ))}
          </ul>
        </div>
      )}
    </main>
  );
}

