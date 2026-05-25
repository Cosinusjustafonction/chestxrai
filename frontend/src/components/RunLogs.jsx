import { useState, useEffect, useRef } from 'react'
import { API_BASE } from '../constants.js'

const TAG_META = {
  '[Queue]':            { label: 'Queue',            color: '#60a5fa' },
  '[Orchestrator]':     { label: 'Orchestrator',     color: '#f59e0b' },
  '[Radiologist]':      { label: 'Radiologist',      color: '#4ade80' },
  '[VLM Review]':       { label: 'VLM Review',       color: '#e879f9' },
  '[Clinical Advisor]': { label: 'Clinical Advisor', color: '#22d3ee' },
  '[Report Generator]': { label: 'Report Generator', color: '#a78bfa' },
  '[Report]':           { label: 'Report',           color: '#a78bfa' },
  '[HITL]':             { label: 'Physician',        color: '#fb923c' },
  '[Error]':            { label: 'Error',            color: '#f87171' },
  '[Warning]':          { label: 'Warning',          color: '#fbbf24' },
}

function getMeta(text) {
  for (const [tag, meta] of Object.entries(TAG_META)) {
    if (text.includes(tag)) return meta
  }
  return { label: 'System', color: '#475569' }
}

function strip(text) {
  return text.replace(/^\[[^\]]+\]\s*/, '').trim()
}

function StatusPill({ status }) {
  const map = {
    queued: '#60a5fa', analyzing: '#60a5fa',
    awaiting_review: '#f59e0b', approved: '#4ade80',
    rejected: '#f87171', error: '#f87171',
  }
  return (
    <span style={{
      fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em',
      padding: '2px 6px', borderRadius: 100,
      background: `${map[status] ?? '#475569'}22`,
      color: map[status] ?? '#475569',
      flexShrink: 0,
    }}>
      {status.replace('_', ' ')}
    </span>
  )
}

function LogEntry({ entry }) {
  const meta = getMeta(entry.text)
  const body = strip(entry.text)
  return (
    <div className="rl-entry">
      <span className="rl-time">{entry.time}</span>
      <span className="rl-label" style={{ color: meta.color }}>{meta.label}</span>
      <span className="rl-body">{body}</span>
    </div>
  )
}

export default function RunLogs({ patients, onClose }) {
  const [selectedId, setSelectedId]   = useState(patients.at(-1)?.id ?? null)
  const [patientData, setPatientData] = useState(null)
  const [loading, setLoading]         = useState(false)
  const logRef = useRef(null)

  useEffect(() => {
    if (!selectedId) return
    setLoading(true)
    fetch(`${API_BASE}/patient/${selectedId}`)
      .then(r => r.json())
      .then(data => { setPatientData(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [selectedId])

  const runLog = patientData?.run_log ?? []
  const report = patientData?.report ?? null

  return (
    <div className="rl-overlay">
      {/* Top bar */}
      <div className="rl-topbar">
        <span className="rl-topbar-title">Agent Run Logs</span>
        <span style={{ color: '#334155', fontSize: 12 }}>
          {runLog.length} entries
        </span>
        <button className="rl-close" onClick={onClose}>✕ Close</button>
      </div>

      <div className="rl-layout">
        {/* Left: run selector */}
        <div className="rl-runs">
          <div className="rl-runs-title">Runs ({patients.length})</div>
          {patients.length === 0 && (
            <p style={{ color: '#334155', fontSize: 12, padding: '8px 12px' }}>
              No runs yet. Upload a scan to start.
            </p>
          )}
          {[...patients].reverse().map(p => (
            <button
              key={p.id}
              className={`rl-run-btn${p.id === selectedId ? ' active' : ''}`}
              onClick={() => setSelectedId(p.id)}
            >
              <span className="rl-run-file" title={p.filename}>{p.filename}</span>
              <StatusPill status={p.status} />
              <span className="rl-run-id">#{p.id}</span>
            </button>
          ))}
        </div>

        {/* Right: log + report */}
        <div className="rl-content">
          {loading && (
            <div className="rl-empty">Loading run log…</div>
          )}
          {!loading && !selectedId && (
            <div className="rl-empty">Select a run on the left.</div>
          )}
          {!loading && selectedId && runLog.length === 0 && (
            <div className="rl-empty">No log entries stored for this run.</div>
          )}

          {!loading && runLog.length > 0 && (
            <>
              {/* Log entries */}
              <div className="rl-section-title">Execution Log</div>
              <div className="rl-log" ref={logRef}>
                {runLog.map((entry, i) => (
                  <LogEntry key={i} entry={entry} />
                ))}
              </div>

              {/* LLM Narrative report */}
              {report && (
                <>
                  <div className="rl-section-title" style={{ marginTop: 24 }}>
                    AI Narrative Report
                    <span style={{ fontSize: 10, fontWeight: 400, color: '#334155', marginLeft: 8 }}>
                      raw LLM output from the crew
                    </span>
                  </div>
                  <pre className="rl-report">{report}</pre>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
