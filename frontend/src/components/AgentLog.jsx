import { useEffect, useRef, useMemo } from 'react'

/* ── Agent display config ─────────────────────────────────── */
const AGENT = {
  't-queue':       { label: 'Queue',            color: '#60a5fa' },
  't-radiologist': { label: 'Radiologist',      color: '#4ade80' },
  't-advisor':     { label: 'Clinical Advisor', color: '#22d3ee' },
  't-report':      { label: 'Report',           color: '#a78bfa' },
  't-hitl':        { label: 'Physician',        color: '#fb923c' },
  't-error':       { label: 'Error',            color: '#f87171' },
  't-warning':     { label: 'Warning',          color: '#fbbf24' },
  't-system':      { label: 'System',           color: '#475569' },
  't-default':     { label: 'System',           color: '#475569' },
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
  const sessions = []   // [ { patientId, filename, time, entries: [] } ]
  let pre = []          // entries before the first dispatch

  for (const entry of logs) {
    const text = entry.text
    // Queue "Dispatching" message marks start of a new patient session
    const dispatch = text.match(/\[Queue\].*Dispatching patient (\S+):\s*(.+)/)
    if (dispatch) {
      sessions.push({
        patientId: dispatch[1],
        filename:  dispatch[2].trim(),
        time:      entry.time,
        entries:   [entry],
      })
    } else if (sessions.length > 0) {
      sessions.at(-1).entries.push(entry)
    } else {
      pre.push(entry)
    }
  }

  return { pre, sessions }
}

/* Group consecutive same-agent entries within a session */
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

/* ── Agent group within a session ────────────────────────── */
function AgentGroup({ group }) {
  const meta = AGENT[group.type] ?? AGENT['t-default']
  const time = group.lines[0].time

  return (
    <div className="al-group" style={{ '--ac': meta.color }}>
      <div className="al-group-head">
        <span className="al-dot" />
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
  )
}

/* ── Patient session card ─────────────────────────────────── */
function PatientSession({ session, index }) {
  const groups = useMemo(() => groupByAgent(session.entries), [session.entries])

  return (
    <div className="al-session">
      <div className="al-session-header">
        <span className="al-session-num">#{index + 1}</span>
        <span className="al-session-file">{session.filename}</span>
        <span className="al-session-id">{session.patientId}</span>
        <span className="al-session-time">{session.time}</span>
      </div>
      <div className="al-session-body">
        {groups.map(g => <AgentGroup key={g.key} group={g} />)}
      </div>
    </div>
  )
}

/* ── Pre-session system messages ─────────────────────────── */
function SystemPre({ entries }) {
  if (entries.length === 0) return null
  const groups = groupByAgent(entries)
  return (
    <div className="al-pre">
      {groups.map(g => <AgentGroup key={g.key} group={g} />)}
    </div>
  )
}

/* ── Main component ──────────────────────────────────────── */
export default function AgentLog({ logs }) {
  const bottomRef = useRef(null)
  const { pre, sessions } = useMemo(() => parseSessions(logs), [logs])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
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

      <div className="al-feed">
        <SystemPre entries={pre} />

        {sessions.length === 0 && pre.length === 0 && (
          <p style={{ color: '#1e3a2a', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
            Waiting for pipeline activity…
          </p>
        )}

        {sessions.map((s, i) => (
          <PatientSession key={s.patientId + s.time} session={s} index={i} />
        ))}

        <div ref={bottomRef} />
      </div>
    </aside>
  )
}
