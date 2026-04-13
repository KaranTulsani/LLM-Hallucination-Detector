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
} from 'lucide-react';

function getScoreConfig(score) {
  if (score >= 75) return {
    label: 'Trustworthy',
    icon: ShieldCheck,
    badgeClass: 'badge-green',
    color: 'var(--clr-green)',
    bgGlow: 'rgba(34, 197, 94, 0.06)',
  };
  if (score >= 50) return {
    label: 'Uncertain',
    icon: ShieldAlert,
    badgeClass: 'badge-yellow',
    color: 'var(--clr-yellow)',
    bgGlow: 'rgba(234, 179, 8, 0.06)',
  };
  return {
    label: 'Likely Hallucinated',
    icon: ShieldX,
    badgeClass: 'badge-red',
    color: 'var(--clr-red)',
    bgGlow: 'rgba(239, 68, 68, 0.06)',
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

  const finalScore = scoreData?.final_score ?? null;
  const config = finalScore !== null ? getScoreConfig(finalScore) : null;
  const Icon = config?.icon;

  return (
    <div
      className="glass-card animate-fade-in-up"
      style={{
        padding: '1.5rem',
        marginBottom: '1rem',
        background: config
          ? `linear-gradient(135deg, ${config.bgGlow}, rgba(18,18,26,0.8))`
          : undefined,
      }}
    >
      {/* ── Query echo ──────────────────────────────────────── */}
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
        {query}
      </p>

      {/* ── Main response ───────────────────────────────────── */}
      <div style={{
        fontSize: '0.95rem',
        lineHeight: '1.7',
        color: 'var(--clr-text)',
        marginBottom: '1.25rem',
        whiteSpace: 'pre-wrap',
      }}>
        {response}
      </div>

      {/* ── Score badge ─────────────────────────────────────── */}
      {config && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '0.75rem',
          marginBottom: '1rem',
        }}>
          <div
            className={`${config.badgeClass} animate-score-reveal`}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.5rem 1rem',
              borderRadius: '2rem',
              fontWeight: 600,
              fontSize: '0.85rem',
            }}
          >
            <Icon size={16} />
            <span>{Math.round(finalScore)}/100</span>
            <span style={{ opacity: 0.7, fontWeight: 400 }}>— {config.label}</span>
          </div>

          {/* Sub-scores pills */}
          <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
            {scoreData?.sub_scores && Object.entries(scoreData.sub_scores).map(([name, val]) => (
              <span
                key={name}
                style={{
                  fontSize: '0.7rem',
                  fontFamily: 'var(--font-mono)',
                  padding: '0.25rem 0.6rem',
                  borderRadius: '1rem',
                  background: 'var(--clr-surface-2)',
                  border: '1px solid var(--clr-border)',
                  color: 'var(--clr-text-dim)',
                }}
              >
                {name.replace(/_/g, ' ')}: {(val * 100).toFixed(0)}%
              </span>
            ))}
          </div>
        </div>
      )}

      {/* ── Penalties ───────────────────────────────────────── */}
      {scoreData?.penalties?.length > 0 && (
        <div style={{
          padding: '0.6rem 0.8rem',
          borderRadius: 'var(--radius)',
          background: 'var(--clr-yellow-dim)',
          border: '1px solid rgba(234, 179, 8, 0.2)',
          marginBottom: '1rem',
          fontSize: '0.82rem',
          color: 'var(--clr-yellow)',
        }}>
          <AlertTriangle size={14} style={{ verticalAlign: 'middle', marginRight: '0.4rem' }} />
          {scoreData.penalties.join(' • ')}
        </div>
      )}

      {/* ── Action buttons ──────────────────────────────────── */}
      {finalScore !== null && (
        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginBottom: showEvidence || showDetectors ? '1rem' : 0 }}>
          <button
            id="show-evidence-btn"
            className="btn btn-ghost"
            onClick={() => setShowEvidence(!showEvidence)}
          >
            <Eye size={15} />
            {showEvidence ? 'Hide' : 'Show'} Evidence
            {showEvidence ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          <button
            id="show-detectors-btn"
            className="btn btn-ghost"
            onClick={() => setShowDetectors(!showDetectors)}
          >
            {showDetectors ? 'Hide' : 'Show'} Detector Details
            {showDetectors ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>

          {!scoreData?.is_trustworthy && !correctedResponse && (
            <button
              id="fix-response-btn"
              className="btn btn-primary"
              onClick={onFix}
              disabled={isFixing}
            >
              {isFixing ? (
                <>
                  <Loader2 size={15} style={{ animation: 'spin 1s linear infinite' }} />
                  Fixing…
                </>
              ) : (
                <>
                  <Wrench size={15} />
                  Fix This Response
                </>
              )}
            </button>
          )}
        </div>
      )}

      {/* ── Evidence panel ──────────────────────────────────── */}
      {showEvidence && (
        <div className="animate-slide-in-left" style={{
          padding: '1rem',
          borderRadius: 'var(--radius)',
          background: 'var(--clr-surface)',
          border: '1px solid var(--clr-border)',
          marginBottom: '1rem',
        }}>
          <h4 style={{
            fontSize: '0.82rem',
            fontWeight: 600,
            color: 'var(--clr-accent)',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            marginBottom: '0.75rem',
          }}>
            Claim Analysis
          </h4>

          {scoreData?.verified_claims?.length > 0 && (
            <div style={{ marginBottom: '0.75rem' }}>
              <div style={{ fontSize: '0.78rem', color: 'var(--clr-green)', fontWeight: 500, marginBottom: '0.4rem' }}>
                ✓ Verified Claims
              </div>
              {scoreData.verified_claims.map((c, i) => (
                <div
                  key={i}
                  style={{
                    padding: '0.4rem 0.6rem',
                    fontSize: '0.82rem',
                    color: 'var(--clr-text-dim)',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '0.4rem',
                    marginBottom: '0.25rem',
                  }}
                >
                  <CheckCircle2 size={14} color="var(--clr-green)" style={{ marginTop: '2px', flexShrink: 0 }} />
                  {c}
                </div>
              ))}
            </div>
          )}

          {scoreData?.unverified_claims?.length > 0 && (
            <div>
              <div style={{ fontSize: '0.78rem', color: 'var(--clr-red)', fontWeight: 500, marginBottom: '0.4rem' }}>
                ✗ Unverified / Flagged Claims
              </div>
              {scoreData.unverified_claims.map((c, i) => (
                <div
                  key={i}
                  style={{
                    padding: '0.4rem 0.6rem',
                    fontSize: '0.82rem',
                    color: 'var(--clr-text-dim)',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: '0.4rem',
                    marginBottom: '0.25rem',
                  }}
                >
                  <XCircle size={14} color="var(--clr-red)" style={{ marginTop: '2px', flexShrink: 0 }} />
                  {c}
                </div>
              ))}
            </div>
          )}

          {scoreData?.evidence?.length > 0 && (
            <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--clr-border)' }}>
              <div style={{ fontSize: '0.78rem', color: 'var(--clr-text-muted)', fontWeight: 500, marginBottom: '0.4rem' }}>
                Evidence Trail
              </div>
              {scoreData.evidence.map((e, i) => (
                <div
                  key={i}
                  style={{
                    padding: '0.3rem 0.6rem',
                    fontSize: '0.78rem',
                    color: 'var(--clr-text-muted)',
                    fontFamily: 'var(--font-mono)',
                    lineHeight: '1.5',
                  }}
                >
                  • {e}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Detector breakdown ──────────────────────────────── */}
      {showDetectors && detectors && (
        <div className="animate-slide-in-left" style={{
          padding: '1rem',
          borderRadius: 'var(--radius)',
          background: 'var(--clr-surface)',
          border: '1px solid var(--clr-border)',
          marginBottom: '1rem',
        }}>
          <h4 style={{
            fontSize: '0.82rem',
            fontWeight: 600,
            color: 'var(--clr-accent)',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            marginBottom: '0.75rem',
          }}>
            Detector Breakdown
          </h4>

          {detectors.map((d, i) => (
            <div
              key={i}
              style={{
                padding: '0.75rem',
                borderRadius: 'var(--radius)',
                background: 'var(--clr-surface-2)',
                border: '1px solid var(--clr-border)',
                marginBottom: '0.5rem',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{
                  fontSize: '0.82rem',
                  fontWeight: 600,
                  color: 'var(--clr-text)',
                  textTransform: 'capitalize',
                }}>
                  {d.name.replace(/_/g, ' ')}
                </span>
                <span style={{
                  fontFamily: 'var(--font-mono)',
                  fontWeight: 600,
                  fontSize: '0.85rem',
                  color: d.score >= 0.7 ? 'var(--clr-green)' : d.score >= 0.4 ? 'var(--clr-yellow)' : 'var(--clr-red)',
                }}>
                  {(d.score * 100).toFixed(0)}%
                </span>
              </div>

              {/* Score bar */}
              <div style={{
                width: '100%',
                height: '4px',
                borderRadius: '2px',
                background: 'var(--clr-border)',
                overflow: 'hidden',
              }}>
                <div style={{
                  width: `${d.score * 100}%`,
                  height: '100%',
                  borderRadius: '2px',
                  background: d.score >= 0.7 ? 'var(--clr-green)' : d.score >= 0.4 ? 'var(--clr-yellow)' : 'var(--clr-red)',
                  transition: 'width 0.8s cubic-bezier(0.19, 1, 0.22, 1)',
                }} />
              </div>

              {d.evidence?.length > 0 && (
                <div style={{ marginTop: '0.5rem' }}>
                  {d.evidence.map((ev, j) => (
                    <div key={j} style={{
                      fontSize: '0.75rem',
                      color: 'var(--clr-text-muted)',
                      fontFamily: 'var(--font-mono)',
                      lineHeight: '1.5',
                    }}>
                      → {ev}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── Corrected response ──────────────────────────────── */}
      {correctedResponse && (
        <div className="animate-fade-in-up" style={{
          padding: '1rem',
          borderRadius: 'var(--radius)',
          background: 'var(--clr-green-dim)',
          border: '1px solid rgba(34, 197, 94, 0.25)',
        }}>
          <div style={{
            fontSize: '0.78rem',
            fontWeight: 600,
            color: 'var(--clr-green)',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            marginBottom: '0.5rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
          }}>
            <Wrench size={14} />
            Corrected Response
          </div>
          <p style={{
            fontSize: '0.92rem',
            lineHeight: '1.7',
            color: 'var(--clr-text)',
            whiteSpace: 'pre-wrap',
          }}>
            {correctedResponse}
          </p>
        </div>
      )}
    </div>
  );
}
