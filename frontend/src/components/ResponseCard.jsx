import { useState } from 'react';
import {
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  ChevronDown,
  ChevronUp,
  Wrench,
  Eye,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Copy,
  Check,
  Globe,
  Brain,
  Zap,
  Activity,
} from 'lucide-react';

function getScoreConfig(score) {
  if (score >= 75) return {
    label: 'Trustworthy',
    icon: ShieldCheck,
    badgeClass: 'badge-green',
    color: '#10b981', // emerald
    bgGlow: 'rgba(16, 185, 129, 0.05)',
  };
  if (score >= 50) return {
    label: 'Uncertain',
    icon: ShieldAlert,
    badgeClass: 'badge-yellow',
    color: '#f59e0b', // amber
    bgGlow: 'rgba(245, 158, 11, 0.05)',
  };
  return {
    label: 'Likely Hallucinated',
    icon: ShieldX,
    badgeClass: 'badge-red',
    color: '#ef4444', // rose
    bgGlow: 'rgba(239, 68, 68, 0.05)',
  };
}

export default function ResponseCard({
  query,
  response,
  scoreData,
  detectors,
  onFix,
  isFixing,
  correctedResponse,
}) {
  const [showEvidence, setShowEvidence] = useState(false);
  const [showDetectors, setShowDetectors] = useState(false);
  const [isCopied, setIsCopied] = useState(false);

  const handleCopy = () => {
    const textToCopy = correctedResponse || response;
    if (textToCopy) {
      navigator.clipboard.writeText(textToCopy);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    }
  };

  const finalScore = scoreData?.final_score ?? null;
  const config = finalScore !== null ? getScoreConfig(finalScore) : null;
  
  // Calculate SVG stroke offset: circumference = 2 * pi * r = 2 * 3.14159 * 40 = 251.3
  const radius = 40;
  const circumference = 2 * Math.PI * radius;
  const strokeOffset = config ? circumference - (finalScore / 100) * circumference : circumference;

  // Compile unique sources and index mapping
  const citations = scoreData?.citations || {};
  const uniqueSources = [];
  const claimMappings = [];

  const getSourceIndex = (url, title) => {
    let idx = uniqueSources.findIndex((s) => s.url === url);
    if (idx === -1) {
      uniqueSources.push({ url, title });
      idx = uniqueSources.length - 1;
    }
    return idx + 1;
  };

  for (const [claim, refs] of Object.entries(citations)) {
    const indices = refs.map((r) => getSourceIndex(r.url, r.title));
    if (indices.length > 0) {
      claimMappings.push({ claim, indices });
    }
  }

  // Sort claims by length descending to match longest matches first
  claimMappings.sort((a, b) => b.claim.length - a.claim.length);

  // Helper to render text with inline claims and footnote indices
  const renderTextWithFootnotes = (text) => {
    if (!text || claimMappings.length === 0) {
      return text;
    }

    let parts = [{ text: text, isClaim: false, indices: [] }];

    for (const mapping of claimMappings) {
      const nextParts = [];
      for (const part of parts) {
        if (part.isClaim) {
          nextParts.push(part);
          continue;
        }

        const index = part.text.toLowerCase().indexOf(mapping.claim.toLowerCase());
        if (index !== -1) {
          const before = part.text.slice(0, index);
          const match = part.text.slice(index, index + mapping.claim.length);
          const after = part.text.slice(index + mapping.claim.length);

          if (before) nextParts.push({ text: before, isClaim: false, indices: [] });
          nextParts.push({ text: match, isClaim: true, indices: mapping.indices });
          if (after) nextParts.push({ text: after, isClaim: false, indices: [] });
        } else {
          nextParts.push(part);
        }
      }
      parts = nextParts;
    }

    return parts.map((part, i) => {
      if (part.isClaim) {
        return (
          <span
            key={i}
            style={{
              borderBottom: '1px dotted rgba(16, 185, 129, 0.4)',
              background: 'rgba(16, 185, 129, 0.03)',
              padding: '0 0.15rem',
              borderRadius: '0.15rem',
            }}
          >
            {part.text}
            {part.indices.map((idx) => (
              <sup key={idx} style={{ fontSize: '0.62rem', fontWeight: 700, marginLeft: '0.06rem', verticalAlign: 'super' }}>
                <a
                  href={uniqueSources[idx - 1].url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: 'var(--clr-green)', textDecoration: 'none', padding: '0 0.05rem' }}
                >
                  [{idx}]
                </a>
              </sup>
            ))}
          </span>
        );
      }
      return part.text;
    });
  };

  return (
    <div
      className="glass-card animate-fade-in-up"
      style={{
        padding: '1.75rem',
        marginBottom: '1.5rem',
        background: config
          ? `linear-gradient(135deg, ${config.bgGlow}, rgba(12,12,20,0.85))`
          : 'var(--clr-surface)',
        border: config ? `1px solid rgba(${config.color === '#10b981' ? '16, 185, 129' : config.color === '#f59e0b' ? '245, 158, 11' : '239, 68, 68'}, 0.15)` : '1px solid var(--clr-border)',
      }}
    >
      {/* ── Header details ──────────────────────────────────── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <div style={{
          fontSize: '0.72rem',
          color: 'var(--clr-text-muted)',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
        }}>
          Evaluation Log
        </div>
        {correctedResponse && (
          <span className="badge badge-green" style={{ fontSize: '0.65rem', padding: '0.2rem 0.5rem' }}>
            Auto-Corrected
          </span>
        )}
      </div>

      <p style={{
        color: 'var(--clr-text-dim)',
        fontSize: '0.9rem',
        marginBottom: '1.25rem',
        padding: '0.75rem 1rem',
        background: 'rgba(255,255,255,0.015)',
        borderRadius: 'var(--radius)',
        borderLeft: '3px solid var(--clr-accent)',
        lineHeight: '1.5',
      }}>
        <strong style={{ color: 'var(--clr-text-muted)', display: 'block', fontSize: '0.7rem', textTransform: 'uppercase', marginBottom: '0.2rem' }}>Prompt</strong>
        {query}
      </p>

      {/* ── Main response text ──────────────────────────────── */}
      <div style={{
        fontSize: '0.98rem',
        lineHeight: '1.75',
        color: 'var(--clr-text)',
        marginBottom: '1.5rem',
        whiteSpace: 'pre-wrap',
        background: 'rgba(0,0,0,0.1)',
        padding: '1rem',
        borderRadius: 'var(--radius)',
        border: '1px solid rgba(255,255,255,0.02)',
      }}>
        {renderTextWithFootnotes(response)}
      </div>

      {/* ── Gauge score & Metadata Section ─────────────────── */}
      {config && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: '80px 1fr',
          gap: '1.25rem',
          alignItems: 'center',
          background: 'rgba(255,255,255,0.01)',
          padding: '1rem',
          borderRadius: 'var(--radius)',
          border: '1px solid rgba(255,255,255,0.02)',
          marginBottom: '1.25rem',
        }}>
          {/* Radial score gauge */}
          <div className="circular-progress" style={{ width: '80px', height: '80px' }}>
            <svg width="80" height="80" viewBox="0 0 100 100">
              <circle
                className="circular-bg"
                cx="50"
                cy="50"
                r={radius}
              />
              <circle
                className="circular-fg"
                cx="50"
                cy="50"
                r={radius}
                stroke={config.color}
                strokeDasharray={circumference}
                strokeDashoffset={strokeOffset}
                style={{ animation: 'gaugeReveal 1s ease-out forwards' }}
              />
            </svg>
            <div style={{
              position: 'absolute',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <span style={{ fontSize: '1.3rem', fontWeight: 800, color: 'var(--clr-text)', fontFamily: 'var(--font-display)', lineHeight: '1' }}>
                {Math.round(finalScore)}
              </span>
              <span style={{ fontSize: '0.5rem', color: 'var(--clr-text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.02em', marginTop: '1px' }}>
                Score
              </span>
            </div>
          </div>

          {/* Details & Pill breakdown */}
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <config.icon size={16} color={config.color} />
              <span style={{
                fontSize: '0.9rem',
                fontWeight: 700,
                color: 'var(--clr-text)',
                fontFamily: 'var(--font-display)',
              }}>
                Verdict: <span style={{ color: config.color }}>{config.label}</span>
              </span>
            </div>
            
            {/* Sub-scores tags list */}
            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
              {scoreData?.sub_scores && Object.entries(scoreData.sub_scores).map(([name, val]) => {
                let sColor = 'var(--clr-text-muted)';
                if (val >= 0.75) sColor = 'var(--clr-green)';
                else if (val >= 0.5) sColor = 'var(--clr-yellow)';
                else sColor = 'var(--clr-red)';
                
                return (
                  <span
                    key={name}
                    style={{
                      fontSize: '0.65rem',
                      fontFamily: 'var(--font-mono)',
                      padding: '0.2rem 0.5rem',
                      borderRadius: '0.5rem',
                      background: 'rgba(255,255,255,0.02)',
                      border: '1px solid var(--clr-border)',
                      color: 'var(--clr-text-dim)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem',
                    }}
                  >
                    <span style={{ width: '5px', height: '5px', borderRadius: '50%', background: sColor }} />
                    {name.replace(/_/g, ' ')}: {(val * 100).toFixed(0)}%
                  </span>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* ── Penalties Banner ────────────────────────────────── */}
      {scoreData?.penalties?.length > 0 && (
        <div style={{
          padding: '0.6rem 0.85rem',
          borderRadius: 'var(--radius)',
          background: 'var(--clr-yellow-dim)',
          border: '1px solid rgba(245, 158, 11, 0.15)',
          marginBottom: '1.25rem',
          fontSize: '0.8rem',
          color: 'var(--clr-yellow)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
        }}>
          <AlertTriangle size={15} style={{ flexShrink: 0 }} />
          <span><strong>Factual Warnings:</strong> {scoreData.penalties.join(' • ')}</span>
        </div>
      )}

      {/* ── Action Panel ────────────────────────────────────── */}
      {finalScore !== null && (
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: showEvidence || showDetectors || correctedResponse ? '1.25rem' : 0 }}>
          <button
            className="btn btn-ghost"
            onClick={handleCopy}
            title="Copy evaluation response to clipboard"
          >
            {isCopied ? <Check size={14} color="var(--clr-green)" /> : <Copy size={14} />}
            {isCopied ? 'Copied Response' : 'Copy'}
          </button>

          <button
            id="show-evidence-btn"
            className="btn btn-ghost"
            onClick={() => setShowEvidence(!showEvidence)}
          >
            <Eye size={14} />
            Evidence Trail
            {showEvidence ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          <button
            id="show-detectors-btn"
            className="btn btn-ghost"
            onClick={() => setShowDetectors(!showDetectors)}
          >
            Detector Details
            {showDetectors ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {!scoreData?.is_trustworthy && !correctedResponse && (
            <button
              id="fix-response-btn"
              className="btn btn-primary"
              onClick={onFix}
              disabled={isFixing}
              style={{ marginLeft: 'auto' }}
            >
              {isFixing ? (
                <>
                  <Loader2 size={14} className="animate-spin" />
                  Applying Fixes…
                </>
              ) : (
                <>
                  <Wrench size={14} />
                  Fix Response
                </>
              )}
            </button>
          )}
        </div>
      )}

      {/* ── Evidence accordion ──────────────────────────────── */}
      {showEvidence && (
        <div className="animate-slide-in-left" style={{
          padding: '1.25rem',
          borderRadius: 'var(--radius)',
          background: 'rgba(255,255,255,0.01)',
          border: '1px solid var(--clr-border)',
          marginBottom: '1.25rem',
        }}>
          <h4 style={{
            fontSize: '0.78rem',
            fontWeight: 700,
            color: 'var(--clr-accent)',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            marginBottom: '0.75rem',
            fontFamily: 'var(--font-display)',
          }}>
            Claim-by-Claim Factual Analysis
          </h4>

          {scoreData?.verified_claims?.length > 0 && (
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--clr-green)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.3rem', marginBottom: '0.5rem' }}>
                <CheckCircle2 size={14} />
                Verified Factual Statements
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                {scoreData.verified_claims.map((c, i) => (
                  <div
                    key={i}
                    style={{
                      padding: '0.5rem 0.75rem',
                      fontSize: '0.82rem',
                      color: 'var(--clr-text-dim)',
                      background: 'rgba(16, 185, 129, 0.02)',
                      border: '1px solid rgba(16, 185, 129, 0.08)',
                      borderRadius: '0.5rem',
                      lineHeight: '1.4',
                    }}
                  >
                    {c}
                  </div>
                ))}
              </div>
            </div>
          )}

          {scoreData?.unverified_claims?.length > 0 && (
            <div style={{ marginBottom: '1rem' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--clr-red)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.3rem', marginBottom: '0.5rem' }}>
                <XCircle size={14} />
                Unverified / Fabricated Claims
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                {scoreData.unverified_claims.map((c, i) => (
                  <div
                    key={i}
                    style={{
                      padding: '0.5rem 0.75rem',
                      fontSize: '0.82rem',
                      color: 'var(--clr-text-dim)',
                      background: 'rgba(239, 68, 68, 0.02)',
                      border: '1px solid rgba(239, 68, 68, 0.08)',
                      borderRadius: '0.5rem',
                      lineHeight: '1.4',
                    }}
                  >
                    {c}
                  </div>
                ))}
              </div>
            </div>
          )}

          {scoreData?.evidence?.length > 0 && (
            <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--clr-border)' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--clr-text-muted)', fontWeight: 600, marginBottom: '0.5rem' }}>
                Web Evidence Trail
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                {scoreData.evidence.map((e, i) => (
                  <div
                    key={i}
                    style={{
                      padding: '0.4rem 0.6rem',
                      fontSize: '0.75rem',
                      color: 'var(--clr-text-muted)',
                      fontFamily: 'var(--font-mono)',
                      background: 'rgba(255,255,255,0.005)',
                      borderRadius: '0.25rem',
                      lineHeight: '1.4',
                    }}
                  >
                    • {e}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Detector details breakdown ───────────────────────── */}
      {showDetectors && detectors && (
        <div className="animate-slide-in-left" style={{
          padding: '1.25rem',
          borderRadius: 'var(--radius)',
          background: 'rgba(255,255,255,0.01)',
          border: '1px solid var(--clr-border)',
          marginBottom: '1.25rem',
        }}>
          <h4 style={{
            fontSize: '0.78rem',
            fontWeight: 700,
            color: 'var(--clr-accent)',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            marginBottom: '0.75rem',
            fontFamily: 'var(--font-display)',
          }}>
            Multi-Signal Score Weighting
          </h4>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {detectors.map((d, i) => {
              const nameMap = {
                'semantic_entropy': { label: 'Semantic Entropy', icon: Brain, color: 'var(--clr-accent-2)' },
                'llm_judge': { label: 'LLM Fact-checking Judge', icon: Zap, color: 'var(--clr-accent)' },
                'rag_grounding': { label: 'RAG Web Search Grounding', icon: Globe, color: 'var(--clr-green)' },
                'nli_entailment': { label: 'NLI Entailment scoring', icon: Activity, color: '#f43f5e' },
              };
              
              const info = nameMap[d.name] || { label: d.name, icon: Activity, color: 'var(--clr-accent)' };
              const DetectorIcon = info.icon;
              
              let statusColor = 'var(--clr-red)';
              if (d.score >= 0.75) statusColor = 'var(--clr-green)';
              else if (d.score >= 0.5) statusColor = 'var(--clr-yellow)';

              return (
                <div
                  key={i}
                  style={{
                    padding: '0.75rem',
                    borderRadius: 'var(--radius)',
                    background: 'var(--clr-surface-2)',
                    border: '1px solid var(--clr-border)',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
                    <span style={{
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      color: 'var(--clr-text)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.4rem',
                    }}>
                      <DetectorIcon size={14} color={info.color} />
                      {info.label}
                    </span>
                    <span style={{
                      fontFamily: 'var(--font-mono)',
                      fontWeight: 700,
                      fontSize: '0.8rem',
                      color: statusColor,
                    }}>
                      {(d.score * 100).toFixed(0)}%
                    </span>
                  </div>

                  {/* Score bar indicator */}
                  <div style={{
                    width: '100%',
                    height: '4px',
                    borderRadius: '2px',
                    background: 'var(--clr-surface-3)',
                    overflow: 'hidden',
                  }}>
                    <div style={{
                      width: `${d.score * 100}%`,
                      height: '100%',
                      borderRadius: '2px',
                      background: statusColor,
                      transition: 'width 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
                    }} />
                  </div>

                  {d.evidence?.length > 0 && (
                    <div style={{ marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
                      {d.evidence.map((ev, j) => (
                        <div key={j} style={{
                          fontSize: '0.72rem',
                          color: 'var(--clr-text-muted)',
                          fontFamily: 'var(--font-mono)',
                          lineHeight: '1.4',
                        }}>
                          → {ev}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Corrected response Output ───────────────────────── */}
      {correctedResponse && (
        <div className="animate-fade-in-up" style={{
          padding: '1.25rem',
          borderRadius: 'var(--radius)',
          background: 'var(--clr-green-dim)',
          border: '1px solid rgba(16, 185, 129, 0.2)',
          marginTop: '1.25rem',
        }}>
          <div style={{
            fontSize: '0.78rem',
            fontWeight: 700,
            color: 'var(--clr-green)',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            marginBottom: '0.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            fontFamily: 'var(--font-display)',
          }}>
            <Wrench size={14} />
            Fact-Corrected Response Output
          </div>
          <p style={{
            fontSize: '0.98rem',
            lineHeight: '1.75',
            color: 'var(--clr-text)',
            whiteSpace: 'pre-wrap',
          }}>
            {renderTextWithFootnotes(correctedResponse)}
          </p>
        </div>
      )}

      {/* ── References & Sources Panel ───────────────────────── */}
      {uniqueSources.length > 0 && (
        <div style={{
          marginTop: '1.5rem',
          paddingTop: '1rem',
          borderTop: '1px solid var(--clr-border)',
        }}>
          <div style={{
            fontSize: '0.72rem',
            color: 'var(--clr-text-muted)',
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            marginBottom: '0.5rem',
            fontFamily: 'var(--font-display)',
          }}>
            Verified References & Citations
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            {uniqueSources.map((src, i) => {
              let hostname = "";
              try {
                hostname = new URL(src.url).hostname;
              } catch (e) {
                hostname = src.url;
              }
              return (
                <div key={i} style={{ fontSize: '0.76rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <span style={{ color: 'var(--clr-green)', fontWeight: 700 }}>[{i + 1}]</span>
                  <a
                    href={src.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: 'var(--clr-text-dim)', textDecoration: 'underline' }}
                  >
                    {src.title}
                  </a>
                  <span style={{ color: 'var(--clr-text-muted)', fontSize: '0.68rem', fontFamily: 'var(--font-mono)' }}>
                    ({hostname})
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
