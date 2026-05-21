function SevBadge({ severity }) {
  if (!severity) return null
  return <span className={`sev-badge ${severity}`}>{severity}</span>
}

function StatusPill({ status }) {
  return <span className={`status-pill sp-${status}`}>{status.replace('_', ' ')}</span>
}

function PatientCard({ patient, isActive, onClick }) {
  const severity = patient.analysis?.severity
  const time = new Date(patient.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

  return (
    <div
      className={`p-card ${severity || 'none'} ${isActive ? 'active' : ''}`}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && onClick()}
    >
      <div className="pc-top">
        <span className="pc-id">#{patient.id}</span>
        <SevBadge severity={severity} />
      </div>
      <div className="pc-file" title={patient.filename}>{patient.filename}</div>
      <div className="pc-meta">
        <span className="pc-time">{time}</span>
        <StatusPill status={patient.status} />
      </div>
    </div>
  )
}

export default function PatientQueue({ patients, selectedId, onSelect, stats }) {
  return (
    <aside className="patient-queue">
      <div className="queue-header">
        <div className="panel-title">Patient Queue</div>
        <div className="queue-stats">
          <div className="stat-cell s-scanned">
            <div className="stat-num">{stats.total}</div>
            <div className="stat-lbl">Scanned</div>
          </div>
          <div className="stat-cell s-review">
            <div className="stat-num">{stats.pending}</div>
            <div className="stat-lbl">Review</div>
          </div>
          <div className="stat-cell s-done">
            <div className="stat-num">{stats.done}</div>
            <div className="stat-lbl">Approved</div>
          </div>
        </div>
      </div>

      {patients.length === 0 ? (
        <div className="queue-empty">
          <div className="queue-empty-icon">🫁</div>
          <div className="queue-empty-text">
            No X-rays in queue.<br />Upload a scan to begin AI analysis.
          </div>
        </div>
      ) : (
        <div className="queue-list">
          {patients.map(p => (
            <PatientCard
              key={p.id}
              patient={p}
              isActive={p.id === selectedId}
              onClick={() => onSelect(p.id)}
            />
          ))}
        </div>
      )}
    </aside>
  )
}
