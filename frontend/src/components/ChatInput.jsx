import { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';

export default function ChatInput({ onSubmit, isLoading }) {
  const [query, setQuery] = useState('');
  const textareaRef = useRef(null);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 160) + 'px';
    }
  }, [query]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;
    onSubmit(query.trim());
    setQuery('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      id="chat-input-form"
      className="relative"
      style={{
        background: 'linear-gradient(135deg, rgba(18,18,26,0.9), rgba(26,26,38,0.7))',
        backdropFilter: 'blur(16px)',
        border: '1px solid var(--clr-border)',
        borderRadius: 'var(--radius-lg)',
        padding: '0.75rem',
        transition: 'border-color 0.3s ease, box-shadow 0.3s ease',
      }}
      onFocus={() => {}}
    >
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0.75rem' }}>
        <textarea
          ref={textareaRef}
          id="chat-query-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything… e.g. 'Who invented the telephone?'"
          rows={1}
          disabled={isLoading}
          style={{
            flex: 1,
            background: 'transparent',
            border: 'none',
            outline: 'none',
            color: 'var(--clr-text)',
            fontSize: '0.95rem',
            fontFamily: 'var(--font-sans)',
            resize: 'none',
            lineHeight: '1.6',
            padding: '0.25rem 0.5rem',
          }}
        />
        <button
          type="submit"
          id="chat-submit-btn"
          disabled={isLoading || !query.trim()}
          className="btn btn-primary"
          style={{
            padding: '0.6rem',
            borderRadius: '0.65rem',
            opacity: isLoading || !query.trim() ? 0.4 : 1,
            minWidth: '42px',
            justifyContent: 'center',
          }}
        >
          {isLoading ? (
            <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
          ) : (
            <Send size={18} />
          )}
        </button>
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginTop: '0.5rem',
          paddingTop: '0.5rem',
          borderTop: '1px solid var(--clr-border)',
        }}
      >
        <span style={{
          fontSize: '0.72rem',
          color: 'var(--clr-text-muted)',
          letterSpacing: '0.02em',
        }}>
          Shift + Enter for new line • Enter to send
        </span>
        <span style={{
          fontSize: '0.72rem',
          color: 'var(--clr-text-muted)',
          fontFamily: 'var(--font-mono)',
        }}>
          {query.length > 0 ? `${query.length} chars` : ''}
        </span>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
        textarea::placeholder {
          color: var(--clr-text-muted);
        }
      `}</style>
    </form>
  );
}
