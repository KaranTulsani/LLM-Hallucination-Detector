import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Sparkles } from 'lucide-react';

export default function ChatInput({ onSubmit, isLoading }) {
  const [query, setQuery] = useState('');
  const [autoFix, setAutoFix] = useState(false);
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
    onSubmit(query.trim(), autoFix);
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
      className="relative glass-card"
      style={{
        padding: '0.85rem',
        border: '1px solid var(--clr-border)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.24)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '0.75rem' }}>
        <textarea
          ref={textareaRef}
          id="chat-query-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a factual question... (e.g. 'Who was the first person on the moon?')"
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
            padding: '0.65rem',
            borderRadius: '0.65rem',
            minWidth: '44px',
            height: '44px',
            justifyContent: 'center',
            display: 'flex',
            alignItems: 'center',
          }}
        >
          {isLoading ? (
            <Loader2 size={18} className="animate-spin" />
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
          marginTop: '0.75rem',
          paddingTop: '0.75rem',
          borderTop: '1px solid var(--clr-border)',
        }}
      >
        {/* Switch component for Auto-Fix */}
        <div 
          className="switch-container" 
          onClick={() => !isLoading && setAutoFix(!autoFix)}
          style={{ opacity: isLoading ? 0.6 : 1 }}
        >
          <div className={`switch-track ${autoFix ? 'active' : ''}`}>
            <div className="switch-thumb" />
          </div>
          <span className="switch-label" style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
            <Sparkles size={12} color={autoFix ? 'var(--clr-accent-2)' : 'var(--clr-text-muted)'} />
            Auto-Fix Hallucinations
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <span style={{
            fontSize: '0.72rem',
            color: 'var(--clr-text-muted)',
            letterSpacing: '0.02em',
          }}>
            Enter to send
          </span>
          {query.length > 0 && (
            <span style={{
              fontSize: '0.72rem',
              color: 'var(--clr-text-muted)',
              fontFamily: 'var(--font-mono)',
              background: 'var(--clr-surface-2)',
              padding: '0.1rem 0.4rem',
              borderRadius: '0.25rem',
            }}>
              {query.length} ch
            </span>
          )}
        </div>
      </div>
    </form>
  );
}
