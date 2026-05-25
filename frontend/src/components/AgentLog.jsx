import { useEffect, useRef, useMemo, useState } from 'react'

/* ── Pipeline step order ──────────────────────────────────── */
const PIPELINE = ['t-queue', 't-orchestrator', 't-radiologist', 't-vlm', 't-advisor', 't-report', 't-hitl']

const AGENT = {
  't-queue':        { label: 'Queue',            color: '#60a5fa', step: 0 },
  't-orchestrator': { label: 'Orchestrator',     color: '#f59e0b', step: 1 },
  't-radiologist':  { label: 'Radiologist',      color: '#4ade80', step: 2 },
  't-vlm':          { label: 'VLM Review',       color: '#e879f9', step: 3 },
  't-advisor':      { label: 'Clinical Advisor', color: '#22d3ee', step: 4 },
  't-report':       { label: 'Report Generator', color: '#a78bfa', step: 5 },
  't-hitl':         { label: 'Physician',        color: '#fb923c', step: 6 },
  't-error':        { label: 'Error',            color: '#f87171', step: -1 },
  't-warning':      { label: 'Warning',          color: '#fbbf24', step: -1 },
  't-system':       { label: 'System',           color: '#475569', step: -1 },
  't-default':      { label: 'System',           color: '#475569', step: -1 },
}

const strip = (text) => text.replace(/^\[[^\]]+\]\s*/, '').trim()

/* Colour keywords inline */
function Hl({ text, color }) {
  const parts = text.split(/([\d.]+%|✦|✓|▶|◦|↳|\bCRITICAL\b|\bABNORMAL\b|\bNORMAL\b|\bDETECTED\b|\bURGENT\b|\bEMERGENT\b|\bROUTINE\b|\bborderline\b)/g)
  return (
    <>
      {parts.map((s, i) => {
        if (/%$/.test(s) || s === '✦' || s === '▶') return <b key={i} style={{ color, fontWeight: 700 }}>{s}</b>
        if (/^(CRITICAL|ABNORMAL|URGENT|EMERGENT)$/.test(s)) return <b key={i} style={{ color: '#f87171' }}>{s}</b>
        if (/^(NORMAL|ROUTINE|DETECTED|✓)$/.test(s))         return <b key={i} style={{ color: '#4ade80' }}>{s}</b>
        if (/^(borderline|◦)$/.test(s))                      return <b key={i} style={{ color: '#fb923c' }}>{s}</b>
        if (s === '↳')                                        return <span key={i} style={{ color: '#334155' }}>{s}</span>
        return s
      })}
    </>
  )
}

/* ── Parse logs into per-patient sessions ─────────────────── */
function parseSessions(logs) {
  const sessions = []
  const pre      = []

  for (const entry of logs) {
    const text     = entry.text
    const dispatch = text.match(/\[Queue\].*Dispatching patient (\S+):\s*(.+)/)
    if (dispatch) {
      sessions.push({ patientId: dispatch[1], filename: dispatch[2].trim(), time: entry.time, entries: [entry] })
    } else if (sessions.length > 0) {
      sessions.at(-1).entries.push(entry)
    } else {
      pre.push(entry)
    }
  }

  return { pre, sessions }
}

/* Group consecutive same-agent entries */
function groupByAgent(entries) {
  const groups = []
  for (const e of entries) {
    const prev = groups.at(-1)
    if (prev && prev.type === e.type) {
      prev.lines.push(e)
    } else {
      groups.push({ key: e.id, type: e.type, lines: [e] })
    }
  }
  return groups
}

/* ── Single agent step block ─────────────────────────────── */
function AgentStep({ group, stepIndex }) {
  const meta    = AGENT[group.type] ?? AGENT['t-default']
  const time    = group.lines[0].time
  const stepNum = meta.step >= 0 ? meta.step + 1 : null

  return (
    <div className="al-step" style={{ '--ac': meta.color }}>
      <div className="al-step-rail">
        <div className="al-step-badge">
          {stepNum !== null ? stepNum : '·'}
        </div>
        <div className="al-step-line" />
      </div>
      <div className="al-step-body">
        <div className="al-step-head">
          <span className="al-agent-name">{meta.label}</span>
          <span className="al-agent-time">{time}</span>
        </div>
        <div className="al-lines">
          {group.lines.map((e, i) => {
            const isLast = i === group.lines.length - 1
            return (
              <div key={e.id} className="al-line">
                <span className="al-tree">{isLast ? '└' : '├'}</span>
                <span className="al-text">
                  <Hl text={strip(e.text)} color={meta.color} />
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

/* ── Patient session card (collapsible) ──────────────────── */
function PatientSession({ session, index, defaultOpen }) {
  const [open, setOpen] = useState(defaultOpen)
  const groups = useMemo(() => groupByAgent(session.entries), [session.entries])

  /* Re-open when new entries arrive in the active session */
  useEffect(() => {
    if (defaultOpen) setOpen(true)
  }, [session.entries.length, defaultOpen])

  const agentTypes = [...new Set(groups.map(g => g.type))]
  const hasVlm     = agentTypes.includes('t-vlm')

  return (
    <div className="al-session">
      <button className="al-session-header" onClick={() => setOpen(o => !o)}>
        <span className="al-session-num">#{index + 1}</span>
        <span className="al-session-file">{session.filename}</span>
        <div className="al-session-dots">
          {PIPELINE.slice(1).map(t => (
            <span
              key={t}
              className="al-pip-dot"
              style={{ background: agentTypes.includes(t) ? AGENT[t].color : 'rgba(255,255,255,.08)' }}
              title={AGENT[t].label}
            />
          ))}
        </div>
        <span className="al-session-id">{session.patientId}</span>
        <span className="al-chevron">{open ? '▾' : '▸'}</span>
      </button>

      {open && (
        <div className="al-session-body">
          {groups.map((g, i) => <AgentStep key={g.key} group={g} stepIndex={i} />)}
        </div>
      )}
    </div>
  )
}

/* ── Pre-session system messages ─────────────────────────── */
function SystemPre({ entries }) {
  if (entries.length === 0) return null
  const groups = groupByAgent(entries)
  return (
    <div className="al-pre">
      {groups.map((g, i) => <AgentStep key={g.key} group={g} stepIndex={i} />)}
    </div>
  )
}

/* ── Main component ──────────────────────────────────────── */
export default function AgentLog({ logs }) {
  const feedRef   = useRef(null)
  const bottomRef = useRef(null)
  const { pre, sessions } = useMemo(() => parseSessions(logs), [logs])

  /* Only auto-scroll when user is already near the bottom */
  useEffect(() => {
    const el = feedRef.current
    if (!el) return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 140
    if (nearBottom) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [sessions.length, sessions.at(-1)?.entries.length])

  return (
    <aside className="agent-log">
      <div className="log-header">
        <span className="log-title">Agent Reasoning</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 10, color: '#1e3a2a', fontFamily: 'var(--font-mono)' }}>
            {sessions.length} scan{sessions.length !== 1 ? 's' : ''}
          </span>
          <div className="log-dot" />
        </div>
      </div>

      <div className="al-feed" ref={feedRef}>
        <SystemPre entries={pre} />

        {sessions.length === 0 && pre.length === 0 && (
          <p style={{ color: '#1e3a2a', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
            Waiting for pipeline activity…
          </p>
        )}

        {sessions.map((s, i) => (
          <PatientSession
            key={s.patientId + s.time}
            session={s}
            index={i}
            defaultOpen={i === sessions.length - 1}
          />
        ))}

        <div ref={bottomRef} />
      </div>
    </aside>
  )
}
