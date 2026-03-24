"use client";
import { useState, useEffect, useRef } from "react";
import axios from "axios";

export default function Home() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState("");
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  
  // Ref for auto-scrolling to bottom
  const scrollRef = useRef(null);

  useEffect(() => {
    setSessionId(generateUUID());
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const generateUUID = () => {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
      var r = (Math.random() * 16) | 0,
        v = c == "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      // ✅ Passing session_id in Headers
      await axios.post("http://localhost:8000/upload", formData, {
        headers: { "session-id": sessionId },
      });
      alert("Document uploaded to this session!");
      setFile(null);
    } catch (e) {
      alert("Upload failed: " + e.message);
    } finally {
      setUploading(false);
    }
  };

  const handleSend = async () => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery) return;

    const userMsg = { role: "user", content: trimmedQuery };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setQuery("");

    // 🛑 Handle "quit" logic
    if (trimmedQuery.toLowerCase() === "quit") {
      try {
        await axios.post("http://localhost:8000/chat", 
          { query: "quit", history: [] }, 
          { headers: { "session-id": sessionId } }
        );
        setMessages([]); // Reset local UI history
        setLoading(false);
        alert("Session Reset: History cleared.");
        return;
      } catch (e) {
        console.error(e);
      }
    }

    try {
      const updatedMessages = [...messages, userMsg];

      const history = updatedMessages.map((m) => ({
        role: m.role,
        content: m.content,
      }));
      // Map current messages to history format for the backend
      //const history = messages.map((m) => ({ role: m.role, content: m.content }));
      
      const res = await axios.post(
        "http://localhost:8000/chat",
        {
          query: userMsg.content,
          history: history, // Send current history
        },
        {
          headers: { "session-id": sessionId }, // ✅ Passing session_id in Headers
        }
      );

      const aiMsg = {
        role: "assistant",
        content: res.data.answer,
        sources: res.data.sources, // This will be empty if "I don't have information"
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error connecting to backend." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    setFile(e.target.files?.[0] || null);
  };

  return (
    <div className="min-h-screen p-8 bg-gradient-to-br from-slate-50 to-slate-200">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-extrabold text-slate-800 mb-2">
            🏗️ Smart RAG Assistant
          </h1>
          <p className="text-slate-600">
            Hybrid Search (BM25 + Vector) • Multi-Query • Session Isolated
          </p>
        </div>

        {/* Upload Section */}
        <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 mb-6">
          <h2 className="text-lg font-semibold text-slate-700 mb-4 flex items-center gap-2">
            <span className="bg-blue-100 p-1 rounded">📁</span> Step 1: Upload Session Documents
          </h2>
          <div className="flex gap-4 items-center">
            <input
              type="file"
              onChange={handleFileChange}
              className="flex-1 border border-slate-300 rounded-lg px-3 py-2 text-sm file:mr-4 file:py-1 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              accept=".txt,.pdf"
            />
            <button
              onClick={handleUpload}
              disabled={uploading || !file}
              className={`px-6 py-2 rounded-lg font-bold transition-all ${
                uploading || !file
                  ? "bg-slate-300 text-slate-500 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-700 text-white shadow-md active:scale-95"
              }`}
            >
              {uploading ? "⏳ Indexing..." : "Upload"}
            </button>
          </div>
          <p className="text-[11px] text-slate-400 mt-2 uppercase tracking-wider font-bold">
            Note: Documents are only searchable in this active session.
          </p>
        </div>

        {/* Chat Section */}
        <div className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 mb-6">
          <div className="flex justify-between items-center mb-4">
             <h2 className="text-lg font-semibold text-slate-700">💬 Step 2: Chat</h2>
             <button 
                onClick={() => {setQuery("quit"); handleSend();}}
                className="text-xs font-bold text-red-500 hover:text-red-700 uppercase"
             >
                Reset Session
             </button>
          </div>
          
          <div 
            ref={scrollRef}
            className="h-[450px] overflow-y-auto border border-slate-100 rounded-lg p-4 mb-4 bg-slate-50 scroll-smooth"
          >
            {messages.length === 0 ? (
              <div className="text-center text-slate-400 mt-20 italic">
                <p>Upload a file to begin.</p>
                <p className="text-xs">Type "quit" to clear history.</p>
              </div>
            ) : (
              messages.map((m, i) => (
                <div key={i} className={`mb-6 ${m.role === "user" ? "text-right" : "text-left"}`}>
                  <div className={`inline-block p-4 rounded-2xl shadow-sm ${
                      m.role === "user" ? "bg-blue-600 text-white rounded-tr-none" : "bg-white border border-slate-200 text-slate-800 rounded-tl-none"
                    } max-w-[85%]`}
                  >
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{m.content}</p>
                    
                    {/* Sources rendered only if present */}
                    {m.sources && m.sources.length > 0 && (
                      <div className="mt-4 pt-3 border-t border-slate-100">
                        <p className="text-[10px] font-bold text-slate-400 uppercase mb-2">Verified Sources</p>
                        <div className="flex flex-wrap gap-2">
                          {m.sources.map((s, idx) => (
                            <details key={idx} className="w-full">
                                <summary className="text-xs text-blue-600 cursor-pointer hover:underline font-medium">
                                   Chunk {idx + 1}: {s.source}
                                </summary>
                                <div className="p-2 mt-1 bg-slate-100 rounded text-[12px] text-slate-600 border-l-2 border-blue-400">
                                   {s.text}
                                </div>
                            </details>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {loading && (
              <div className="flex gap-2 items-center text-slate-400 animate-pulse">
                <div className="w-2 h-2 bg-slate-400 rounded-full"></div>
                <p className="text-xs font-medium">Processing Multi-Query RAG...</p>
              </div>
            )}
          </div>

          <div className="flex gap-3">
            <input
              className="flex-1 border border-slate-300 p-3 rounded-xl focus:ring-2 focus:ring-blue-500 outline-none transition-all"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Ask a question or type 'quit'..."
              disabled={loading}
            />
            <button
              onClick={handleSend}
              disabled={loading || !query.trim()}
              className="bg-slate-800 hover:bg-black text-white px-6 py-3 rounded-xl font-bold transition-all disabled:bg-slate-300"
            >
              Ask
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-between items-center px-2">
            <div className="text-[10px] text-slate-400 font-mono">
                ID: {sessionId}
            </div>
            <div className="text-[10px] text-slate-400 font-bold uppercase tracking-widest">
                Local RAG Engine v2.0
            </div>
        </div>
      </div>
    </div>
  );
}