import { useState, useRef, useEffect } from 'react';
import {
  Shield,
  Zap,
  Brain,
  Globe,
  Activity,
  ArrowRight,
  Terminal,
  Cpu,
  RotateCcw,
  Sparkles,
  Server,
  HeartPulse,
} from 'lucide-react';
import ChatInput from './components/ChatInput';
import ResponseCard from './components/ResponseCard';
import { chatStreamWithDetection, correctResponse, healthCheck } from './api';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [fixingId, setFixingId] = useState(null);
  const [loadingStep, setLoadingStep] = useState('');
  const [backendHealth, setBackendHealth] = useState({ status: 'connecting', model: '', judge_model: '' });
  const scrollRef = useRef(null);

  // Fetch backend health on mount
  useEffect(() => {
    async function checkHealth() {
      try {
        const data = await healthCheck();
        setBackendHealth({
          status: 'online',
          model: data.model || 'llama-3.1-8b-instant',
          judge_model: data.judge_model || 'llama-3.3-70b-versatile',
        });
      } catch (err) {
        setBackendHealth({
          status: 'offline',
          model: 'Unknown',
          judge_model: 'Unknown',
        });
      }
    }
    checkHealth();
  }, []);

  // Auto-scroll on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [messages]);

  const handleSubmit = async (query, autoFix = false) => {
    const id = Date.now();
    
    // Compile history
    const history = messages
      .filter((m) => !m.loading && m.response)
      .flatMap((m) => [
        { role: 'user', content: m.query },
        { role: 'assistant', content: m.corrected || m.response },
      ]);

    setMessages((prev) => [
      ...prev,
      { id, query, response: null, scoreData: null, detectors: null, corrected: null, loading: true },
    ]);
    setIsLoading(true);
    
    // Simulate multi-step loading process for transparency
    setLoadingStep('Generating LLM response...');
    
    try {
      const stream = chatStreamWithDetection(query, null, autoFix, history.length > 0 ? history : null);
      
      for await (const data of stream) {
        if (data.chunk !== undefined) {
          setLoadingStep('Receiving stream...');
          setMessages((prev) =>
            prev.map((m) =>
              m.id === id
                ? { ...m, response: (m.response || "") + data.chunk }
                : m
            )
          );
        } else if (data.result !== undefined) {
          setLoadingStep('Running 4 independent detectors...');
          setMessages((prev) =>
            prev.map((m) =>
              m.id === id
                ? {
                    ...m,
                    response: data.result.response,
                    scoreData: data.result.score,
                    detectors: data.result.detectors,
                    corrected: data.result.corrected_response || null,
                    loading: false,
                  }
                : m
            )
          );
        } else if (data.error !== undefined) {
          throw new Error(data.error);
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === id
            ? {
                ...m,
                response: `Error: ${err.message}. Ensure backend is running locally on port 8000.`,
                loading: false,
              }
            : m
        )
      );
    } finally {
      setIsLoading(false);
      setLoadingStep('');
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
      console.error('Correction failed:', err);
    } finally {
      setFixingId(null);
    }
  };

  const resetChat = () => {
    if (window.confirm("Are you sure you want to clear the conversation?")) {
      setMessages([]);
    }
  };

  // Preset query list
  const suggestionQueries = [
    { text: "When was the Eiffel Tower built?", desc: "Check if year matches historical records" },
    { text: "Who invented the telephone?", desc: "Evaluate multiple claimants consistency" },
    { text: "Explain the safety of honey bee venom.", desc: "Verify scientific and medical claims" },
    { text: "Who is the CEO of Apple Inc.?", desc: "Test current standard factual data search" },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', background: 'var(--clr-bg)' }}>
      {/* ── Main Container Grid ──────────────────────────────── */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(280px, 320px) 1fr',
        minHeight: '100vh',
        width: '100%',
      }} className="responsive-grid">
        
        {/* ── Sidebar Column ─────────────────────────────────── */}
        <aside style={{
          borderRight: '1px solid var(--clr-border)',
          background: 'rgba(6, 6, 10, 0.95)',
          padding: '1.5rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.5rem',
          position: 'sticky',
          top: 0,
          height: '100vh',
          zIndex: 10,
        }} className="sidebar-container">
          
          {/* Brand Logo */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              width: '40px',
              height: '40px',
              borderRadius: '12px',
              background: 'linear-gradient(135deg, var(--clr-accent), var(--clr-accent-2))',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 4px 16px rgba(99, 102, 241, 0.25)',
            }}>
              <Shield size={22} color="#fff" />
            </div>
            <div>
              <h1 style={{
                fontSize: '1.2rem',
                fontWeight: 800,
                letterSpacing: '-0.02em',
                color: 'var(--clr-text)',
                lineHeight: '1.1',
              }}>
                Verify.io
              </h1>
              <p style={{
                fontSize: '0.68rem',
                color: 'var(--clr-text-muted)',
                fontWeight: 600,
                letterSpacing: '0.04em',
                textTransform: 'uppercase',
              }}>
                Hallucination Guard
              </p>
            </div>
          </div>

          {/* System Health Status */}
          <div className="glass-card" style={{ padding: '1rem', background: 'rgba(255,255,255,0.015)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--clr-text-dim)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                System Status
              </span>
              <span style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.3rem',
                fontSize: '0.65rem',
                fontWeight: 700,
                color: backendHealth.status === 'online' ? 'var(--clr-green)' : backendHealth.status === 'offline' ? 'var(--clr-red)' : 'var(--clr-yellow)',
                textTransform: 'uppercase',
              }}>
                <HeartPulse size={12} className={backendHealth.status === 'connecting' ? 'animate-pulse' : ''} />
                {backendHealth.status}
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.75rem' }}>
              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                <Cpu size={12} color="var(--clr-accent)" />
                <span style={{ color: 'var(--clr-text-muted)' }}>Response:</span>
                <span style={{ color: 'var(--clr-text-dim)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                  {backendHealth.model || 'Loading...'}
                </span>
              </div>
              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                <Server size={12} color="var(--clr-accent-2)" />
                <span style={{ color: 'var(--clr-text-muted)' }}>Judge:</span>
                <span style={{ color: 'var(--clr-text-dim)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                  {backendHealth.judge_model || 'Loading...'}
                </span>
              </div>
            </div>
          </div>

          {/* Configuration Settings */}
          <div className="glass-card" style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem', background: 'rgba(255,255,255,0.015)' }}>
            <h3 style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--clr-text-dim)' }}>
              Active Evaluators
            </h3>

            {/* Active signals metrics info */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {[
                { icon: Globe, label: 'Web Grounding', desc: 'RAG verification via Serper', active: true, color: 'var(--clr-green)' },
                { icon: Zap, label: 'LLM Judge Fact-check', desc: 'Structured analysis (llama-3.3)', active: true, color: 'var(--clr-accent)' },
                { icon: Brain, label: 'Semantic Entropy', desc: 'Self-consistency embeddings', active: true, color: 'var(--clr-accent-2)' },
                { icon: Activity, label: 'NLI Entailment', desc: 'Natural language alignment', active: true, color: '#f43f5e' },
              ].map((sig, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.6rem' }}>
                  <sig.icon size={15} color={sig.color} style={{ marginTop: '2px' }} />
                  <div>
                    <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--clr-text)' }}>{sig.label}</div>
                    <div style={{ fontSize: '0.68rem', color: 'var(--clr-text-muted)' }}>{sig.desc}</div>
                  </div>
                  <div style={{
                    marginLeft: 'auto',
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    background: sig.active ? 'var(--clr-green)' : 'var(--clr-text-muted)',
                    boxShadow: sig.active ? '0 0 8px var(--clr-green)' : 'none',
                    alignSelf: 'center',
                  }} />
                </div>
              ))}
            </div>

            <div style={{ borderTop: '1px solid var(--clr-border)', paddingTop: '0.75rem', fontSize: '0.75rem', color: 'var(--clr-text-muted)' }}>
              <span>Score Threshold: <strong>65/100</strong></span>
              <div style={{ width: '100%', height: '4px', background: 'var(--clr-surface-2)', borderRadius: '2px', marginTop: '0.25rem', position: 'relative' }}>
                <div style={{ width: '65%', height: '100%', background: 'var(--clr-accent)', borderRadius: '2px' }} />
                <div style={{ position: 'absolute', width: '8px', height: '8px', borderRadius: '50%', background: '#fff', top: '-2px', left: '65%', transform: 'translateX(-50%)', border: '1px solid var(--clr-accent)' }} />
              </div>
            </div>
          </div>

          {/* Reset Buttons */}
          <div style={{ marginTop: 'auto' }}>
            <button
              onClick={resetChat}
              disabled={messages.length === 0}
              className="btn btn-ghost"
              style={{
                width: '100%',
                justifyContent: 'center',
                gap: '0.4rem',
                fontSize: '0.78rem',
                opacity: messages.length === 0 ? 0.4 : 1,
              }}
            >
              <RotateCcw size={14} />
              Reset Conversation
            </button>
          </div>
        </aside>

        {/* ── Chat Vewport Column ────────────────────────────── */}
        <section style={{
          display: 'flex',
          flexDirection: 'column',
          height: '100vh',
          overflow: 'hidden',
          position: 'relative',
        }} className="chat-container">

          {/* Header Panel */}
          <header style={{
            padding: '1rem 1.5rem',
            background: 'rgba(6, 6, 10, 0.8)',
            backdropFilter: 'blur(16px)',
            borderBottom: '1px solid var(--clr-border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            zIndex: 5,
          }}>
            <div>
              <h2 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--clr-text)' }}>Evaluator Console</h2>
              <p style={{ fontSize: '0.7rem', color: 'var(--clr-text-muted)' }}>Run claims through double validation models</p>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span className="badge badge-green" style={{ fontSize: '0.62rem', letterSpacing: '0.04em' }}>Web Grounding Active</span>
            </div>
          </header>

          {/* Scrollable Chat Area */}
          <div
            ref={scrollRef}
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: '1.5rem',
            }}
          >
            <div style={{ maxWidth: '820px', margin: '0 auto', width: '100%' }}>
              
              {/* Landing view when empty */}
              {messages.length === 0 && (
                <div
                  className="animate-fade-in-up"
                  style={{
                    textAlign: 'center',
                    paddingTop: '6vh',
                    paddingBottom: '3rem',
                  }}
                >
                  <div style={{
                    width: '64px',
                    height: '64px',
                    borderRadius: '16px',
                    background: 'var(--clr-accent-dim)',
                    border: '1px solid rgba(99, 102, 241, 0.2)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto 1.25rem',
                    animation: 'pulse-ring 2.5s infinite',
                  }}>
                    <Shield size={30} color="var(--clr-accent)" />
                  </div>

                  <h2 style={{
                    fontSize: '1.75rem',
                    fontWeight: 800,
                    marginBottom: '0.5rem',
                    letterSpacing: '-0.02em',
                    background: 'linear-gradient(135deg, var(--clr-text), var(--clr-accent-2))',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                  }}>
                    Factual Hallucination Guard
                  </h2>

                  <p style={{
                    color: 'var(--clr-text-dim)',
                    fontSize: '0.9rem',
                    maxWidth: '480px',
                    margin: '0 auto 2.25rem',
                    lineHeight: '1.6',
                  }}>
                    Ask factual queries. We'll run them against a live Google Web search retriever, extract individual claims, and run dual-model cross-evaluations.
                  </p>

                  {/* Suggestions Queries Cards Grid */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: '0.75rem',
                    maxWidth: '640px',
                    margin: '0 auto',
                  }} className="suggestion-grid">
                    {suggestionQueries.map((item, i) => (
                      <div
                        key={i}
                        className="suggestion-card"
                        onClick={() => handleSubmit(item.text)}
                      >
                        <div style={{ textAlign: 'left' }}>
                          <div style={{ fontWeight: 600, color: 'var(--clr-text)' }}>{item.text}</div>
                          <div style={{ fontSize: '0.68rem', color: 'var(--clr-text-muted)', marginTop: '0.15rem' }}>{item.desc}</div>
                        </div>
                        <ArrowRight size={14} style={{ flexShrink: 0, opacity: 0.6 }} />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Message log */}
              {messages.map((msg) => (
                <div key={msg.id}>
                  {msg.loading ? (
                    <div
                      className="glass-card animate-fade-in-up"
                      style={{ padding: '1.5rem', marginBottom: '1rem', background: 'var(--clr-surface)' }}
                    >
                      <div style={{
                        fontSize: '0.72rem',
                        color: 'var(--clr-text-muted)',
                        marginBottom: '0.5rem',
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        letterSpacing: '0.06em',
                      }}>
                        Prompt
                      </div>
                      <p style={{
                        color: 'var(--clr-text-dim)',
                        fontSize: '0.9rem',
                        marginBottom: '1.25rem',
                        padding: '0.75rem 1rem',
                        background: 'rgba(255,255,255,0.015)',
                        borderRadius: 'var(--radius)',
                        borderLeft: '3px solid var(--clr-accent)',
                      }}>
                        {msg.query}
                      </p>

                      {msg.response && (
                        <div style={{
                          fontSize: '0.98rem',
                          lineHeight: '1.75',
                          color: 'var(--clr-text)',
                          marginBottom: '1.5rem',
                          whiteSpace: 'pre-wrap',
                          background: 'rgba(0,0,0,0.1)',
                          padding: '1rem',
                          borderRadius: 'var(--radius)',
                        }}>
                          {msg.response}
                        </div>
                      )}

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        {!msg.response && (
                          <>
                            <div className="shimmer" style={{ height: '16px', width: '95%' }} />
                            <div className="shimmer" style={{ height: '16px', width: '80%' }} />
                          </>
                        )}
                        <div className="shimmer" style={{ height: '16px', width: '60%' }} />
                      </div>

                      <p style={{
                        fontSize: '0.76rem',
                        color: 'var(--clr-accent)',
                        marginTop: '1.25rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.45rem',
                        fontWeight: 600,
                        fontFamily: 'var(--font-display)',
                      }}>
                        <Activity size={14} className="animate-pulse" />
                        <span>{loadingStep || "Analyzing claim consistency..."}</span>
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
            </div>
          </div>

          {/* Sticky Input Area */}
          <div style={{
            padding: '1rem 1.5rem 1.5rem',
            background: 'linear-gradient(to top, var(--clr-bg) 60%, transparent)',
            borderTop: '1px solid transparent',
            zIndex: 5,
          }}>
            <div style={{ maxWidth: '820px', margin: '0 auto', width: '100%' }}>
              <ChatInput onSubmit={handleSubmit} isLoading={isLoading} />
            </div>
          </div>
        </section>

      </div>

      {/* Responsive layout fallback helper style */}
      <style>{`
        @media (max-width: 768px) {
          .responsive-grid {
            grid-template-columns: 1fr !important;
          }
          .sidebar-container {
            height: auto !important;
            border-right: none !important;
            border-bottom: 1px solid var(--clr-border) !important;
            position: relative !important;
          }
          .chat-container {
            height: calc(100vh - 250px) !important;
          }
          .suggestion-grid {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </div>
  );
}
