import { FormEvent, useEffect, useRef, useState } from "react";
import { streamChat } from "@/api/client";
import type { ChatMessage } from "@/api/types";
import { useAppStore } from "@/store/useAppStore";

interface ChatPanelProps {
  /** "dashboard" for the field-wide chat, or a well_id for a well-scoped chat. */
  scope: string;
  wellId: string | null;
  title: string;
  subtitle: string;
}

interface ToolCallRecord {
  name: string;
  input: Record<string, unknown>;
  output: unknown;
}

/**
 * Persistent light-themed chat panel. Streams tokens from the backend's
 * /chat SSE endpoint and shows a transparent, collapsible log of any tool
 * calls the agent made -- so the user can see exactly which real computed
 * values grounded the answer (per section 5 of the brief).
 */
export default function ChatPanel({ scope, wellId, title, subtitle }: ChatPanelProps) {
  const { chatOpen, toggleChat, conversations, appendMessage } = useAppStore();
  const messages = conversations[scope] ?? [];

  const [input, setInput] = useState("");
  const [streamingText, setStreamingText] = useState("");
  const [pendingTools, setPendingTools] = useState<ToolCallRecord[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, streamingText, pendingTools]);

  useEffect(() => () => abortRef.current?.(), []);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;

    const userMessage: ChatMessage = { role: "user", content: trimmed };
    appendMessage(scope, userMessage);
    setInput("");
    setError(null);
    setStreamingText("");
    setPendingTools([]);
    setIsStreaming(true);

    const history = messages.map((m) => ({ role: m.role, content: m.content }));

    let fullText = "";
    abortRef.current = streamChat(
      trimmed,
      wellId,
      history,
      (event) => {
        if (event.type === "text_delta") {
          fullText += event.text;
          setStreamingText(fullText);
        } else if (event.type === "tool_call") {
          setPendingTools((prev) => [...prev, event]);
        } else if (event.type === "done") {
          appendMessage(scope, { role: "assistant", content: fullText });
          setStreamingText("");
          setIsStreaming(false);
        } else if (event.type === "error") {
          setError(event.message);
          setIsStreaming(false);
        }
      },
      (err) => {
        setError(err.message);
        setIsStreaming(false);
      }
    );
  }

  if (!chatOpen) {
    return (
      <button
        onClick={toggleChat}
        className="fixed bottom-6 right-6 z-40 rounded-full bg-accent text-white shadow-lg px-5 py-3 text-sm font-medium hover:bg-accent-strong transition-colors"
      >
        Ask the assistant
      </button>
    );
  }

  return (
    <aside className="fixed top-0 right-0 h-full w-[380px] bg-surface border-l border-border shadow-xl z-40 flex flex-col">
      <div className="px-4 py-3 border-b border-border flex items-start justify-between">
        <div>
          <h3 className="font-semibold text-sm">{title}</h3>
          <p className="text-xs text-ink-faint">{subtitle}</p>
        </div>
        <button
          onClick={toggleChat}
          className="text-ink-faint hover:text-ink text-lg leading-none px-1"
          aria-label="Close chat"
        >
          ×
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3 bg-surface-muted">
        {messages.length === 0 && !streamingText && (
          <p className="text-xs text-ink-faint leading-relaxed">
            Ask about VSH, PHIE, SWE, zonation, or compare wells. Answers are grounded in the
            computed curves for this dataset -- the assistant will flag when a cutoff or
            assumption (Rw, Swirr, matrix density) may need SME review.
          </p>
        )}

        {messages.map((m, i) => (
          <ChatBubble key={i} role={m.role} content={m.content} />
        ))}

        {pendingTools.length > 0 && (
          <div className="space-y-1">
            {pendingTools.map((t, i) => (
              <ToolCallChip key={i} record={t} />
            ))}
          </div>
        )}

        {streamingText && <ChatBubble role="assistant" content={streamingText} />}
        {isStreaming && !streamingText && (
          <div className="text-xs text-ink-faint italic px-1">Thinking…</div>
        )}
        {error && (
          <div className="text-xs text-danger bg-red-50 border border-red-200 rounded-md px-3 py-2">
            {error}
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="p-3 border-t border-border bg-surface">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question…"
            className="flex-1 rounded-md border border-border-strong bg-surface px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-accent/40 focus:border-accent"
            disabled={isStreaming}
          />
          <button
            type="submit"
            disabled={isStreaming || !input.trim()}
            className="rounded-md bg-accent text-white text-sm font-medium px-3 py-2 disabled:opacity-40 hover:bg-accent-strong transition-colors"
          >
            Send
          </button>
        </div>
      </form>
    </aside>
  );
}

function ChatBubble({ role, content }: { role: string; content: string }) {
  const isUser = role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap leading-relaxed ${
          isUser
            ? "bg-accent text-white rounded-br-sm"
            : "bg-surface border border-border text-ink rounded-bl-sm"
        }`}
      >
        {content}
      </div>
    </div>
  );
}

function ToolCallChip({ record }: { record: ToolCallRecord }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="text-xs border border-border rounded-md bg-surface">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full text-left px-2 py-1.5 flex items-center gap-1.5 text-ink-muted hover:bg-surface-sunken rounded-md"
      >
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent" />
        Called <code className="font-mono">{record.name}</code>
        <span className="ml-auto text-ink-faint">{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <pre className="px-2 pb-2 text-[11px] text-ink-muted overflow-x-auto">
          {JSON.stringify({ input: record.input, output: record.output }, null, 2)}
        </pre>
      )}
    </div>
  );
}
