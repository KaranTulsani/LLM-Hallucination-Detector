import { useState, useRef, useEffect } from 'react';
import {
  Shield,
  Zap,
  Brain,
  Database,
  Activity,
} from 'lucide-react';
import ChatInput from './components/ChatInput';
import ResponseCard from './components/ResponseCard';
import { chatWithDetection, correctResponse } from './api';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [fixingId, setFixingId] = useState(null);
  const scrollRef = useRef(null);

  // Auto-scroll on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages]);

  const handleSubmit = async (query) => {
    const id = Date.now();
    setMessages((prev) => [
      ...prev,
      { id, query, response: null, scoreData: null, detectors: null, corrected: null, loading: true },
    ]);
    setIsLoading(true);

    try {
      const data = await chatWithDetection(query);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? {
                ...m,
                response: data.response,
                scoreData: data.score,
                detectors: data.detectors,
                corrected: data.corrected_response || null,
                loading: false,
              }
            : m
        )
      );
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? {
                ...m,
                response: `Error: ${err.message}. Make sure the backend is running on port 8000.`,
                loading: false,
              }
            : m
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleFix = async (msg) => {
    setFixingId(msg.id);
    try {
      const data = await correctResponse(
        msg.query,
        msg.response,
        msg.scoreData?.unverified_claims || [],
        'constrained'
      );
      setMessages((prev) =>
        prev.map((m) =>
          m.id === msg.id ? { ...m, corrected: data.corrected_response } : m
        )
      );
    } catch (err) {
      console.error('Fix failed:', err);
    } finally {
      setFixingId(null);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* ── Header ──────────────────────────────────────────── */}
      <header
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 50,
          padding: '0.85rem 1.5rem',
          background: 'rgba(10, 10, 15, 0.85)',
          backdropFilter: 'blur(16px)',
          borderBottom: '1px solid var(--clr-border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, var(--clr-accent), var(--clr-accent-2))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 2px 12px rgba(129, 140, 248, 0.3)',
          }}>
            <Shield size={20} color="#fff" />
          </div>
          <div>
            <h1 style={{
              fontSize: '1.05rem',
              fontWeight: 700,
              letterSpacing: '-0.01em',
              background: 'linear-gradient(135deg, var(--clr-text), var(--clr-accent))',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              Hallucination Detector
            </h1>
            <p style={{
              fontSize: '0.68rem',
              color: 'var(--clr-text-muted)',
              letterSpacing: '0.03em',
            }}>
              Multi-signal LLM trustworthiness scoring
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <span style={{
            fontSize: '0.7rem',
            fontFamily: 'var(--font-mono)',
            padding: '0.25rem 0.6rem',
            borderRadius: '1rem',
            background: 'var(--clr-surface-2)',
            border: '1px solid var(--clr-border)',
            color: 'var(--clr-text-dim)',
          }}>
            v1.0
          </span>
        </div>
      </header>

      {/* ── Main content ────────────────────────────────────── */}
      <main
        ref={scrollRef}
        style={{
          flex: 1,
          overflow: 'auto',
          padding: '1.5rem',
          maxWidth: '860px',
          width: '100%',
          margin: '0 auto',
        }}
      >
        {/* Empty state */}
        {messages.length === 0 && (
          <div
            className="animate-fade-in-up"
            style={{
              textAlign: 'center',
              paddingTop: '8vh',
              paddingBottom: '4rem',
            }}
          >
            <div style={{
              width: '72px',
              height: '72px',
              borderRadius: '18px',
              background: 'linear-gradient(135deg, var(--clr-accent-dim), rgba(167,139,250,0.12))',
              border: '1px solid rgba(129,140,248,0.2)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 1.5rem',
              animation: 'pulse-ring 3s ease-in-out infinite',
            }}>
              <Shield size={34} color="var(--clr-accent)" />
            </div>

            <h2 style={{
              fontSize: '1.6rem',
              fontWeight: 700,
              marginBottom: '0.5rem',
              background: 'linear-gradient(135deg, var(--clr-text), var(--clr-accent))',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}>
              AI Hallucination Detector
            </h2>

            <p style={{
              color: 'var(--clr-text-dim)',
              fontSize: '0.95rem',
              maxWidth: '480px',
              margin: '0 auto 2.5rem',
              lineHeight: '1.6',
            }}>
              Ask any question. We'll generate a response, run three independent
              detectors, and score its trustworthiness from 0 to 100.
            </p>

            {/* Feature cards */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '1rem',
              maxWidth: '680px',
              margin: '0 auto',
            }}>
              {[
                { icon: Brain, title: 'Semantic Entropy', desc: 'Multi-sample consistency via embeddings' },
                { icon: Zap, title: 'LLM Judge', desc: 'Structured fact-checking prompt' },
                { icon: Database, title: 'RAG Grounding', desc: 'Knowledge-base verification' },
                { icon: Activity, title: 'Smart Scoring', desc: 'Weighted aggregation + penalties' },
              ].map((f, i) => (
                <div
                  key={i}
                  className="glass-card"
                  style={{
                    padding: '1.25rem',
                    textAlign: 'left',
                    animationDelay: `${i * 0.1}s`,
                  }}
                >
                  <f.icon size={20} color="var(--clr-accent)" style={{ marginBottom: '0.6rem' }} />
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.25rem' }}>
                    {f.title}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--clr-text-muted)', lineHeight: '1.4' }}>
                    {f.desc}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Message list */}
        {messages.map((msg) => (
          <div key={msg.id}>
            {msg.loading ? (
              <div
                className="glass-card animate-fade-in-up"
                style={{ padding: '1.5rem', marginBottom: '1rem' }}
              >
                <div style={{
                  fontSize: '0.8rem',
                  color: 'var(--clr-text-muted)',
                  marginBottom: '0.5rem',
                  fontWeight: 500,
                  textTransform: 'uppercase',
                  letterSpacing: '0.06em',
                }}>
                  You asked
                </div>
                <p style={{
                  color: 'var(--clr-text-dim)',
                  fontSize: '0.92rem',
                  marginBottom: '1.25rem',
                  padding: '0.75rem',
                  background: 'rgba(255,255,255,0.02)',
                  borderRadius: 'var(--radius)',
                  borderLeft: '3px solid var(--clr-accent)',
                }}>
                  {msg.query}
                </p>

                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.75rem',
                }}>
                  <div className="shimmer" style={{ height: '16px', width: '90%' }} />
                  <div className="shimmer" style={{ height: '16px', width: '75%' }} />
                  <div className="shimmer" style={{ height: '16px', width: '60%' }} />
                  <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
                    <div className="shimmer" style={{ height: '28px', width: '120px' }} />
                    <div className="shimmer" style={{ height: '28px', width: '80px' }} />
                  </div>
                </div>

                <p style={{
                  fontSize: '0.78rem',
                  color: 'var(--clr-text-muted)',
                  marginTop: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                }}>
                  <Activity size={13} style={{ animation: 'pulse-ring 1.5s ease-in-out infinite' }} />
                  Generating response & running 3 detectors in parallel…
                </p>
              </div>
            ) : (
              <ResponseCard
                query={msg.query}
                response={msg.response}
                scoreData={msg.scoreData}
                detectors={msg.detectors}
                onFix={() => handleFix(msg)}
                isFixing={fixingId === msg.id}
                correctedResponse={msg.corrected}
              />
            )}
          </div>
        ))}
      </main>

      {/* ── Input area ──────────────────────────────────────── */}
      <div style={{
        position: 'sticky',
        bottom: 0,
        padding: '1rem 1.5rem 1.25rem',
        maxWidth: '860px',
        width: '100%',
        margin: '0 auto',
        background: 'linear-gradient(to top, var(--clr-bg) 60%, transparent)',
      }}>
        <ChatInput onSubmit={handleSubmit} isLoading={isLoading} />
      </div>
    </div>
  );
}
