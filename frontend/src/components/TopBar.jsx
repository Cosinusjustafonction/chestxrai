import { useRef } from 'react'

function TbChip({ children, cls }) {
  return <span className={`tb-chip ${cls}`}>{children}</span>
}

export default function TopBar({
  onUpload, onApprove, onReject, onRevise,
  canReview, isUploading, selectedPatient,
  reviewNotes, setReviewNotes,
}) {
  const fileRef = useRef(null)

  const handleFile = (e) => {
    const f = e.target.files?.[0]
    if (f) { onUpload(f); e.target.value = '' }
  }

  const p = selectedPatient

  return (
    <header className="topbar">
      {/* Brand */}
      <div className="topbar-brand">
        <div className="brand-icon">🫁</div>
        <div>
          <div className="brand-name">ChestXRAI</div>
          <div className="brand-sub">Radiology Triage System</div>
        </div>
      </div>

      <div className="topbar-sep" />

      {/* Context */}
      <div className="topbar-center">
        {p ? (
          <>
            <span className="tb-label">Active</span>
            <span className="tb-id">#{p.id}</span>
            <TbChip cls={p.status}>{p.status.replace('_', ' ')}</TbChip>
            {p.analysis?.severity && (
              <TbChip cls={p.analysis.severity}>{p.analysis.severity}</TbChip>
            )}
            {p.analysis?.top_finding && (
              <span style={{ fontSize: 12, color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                — {p.analysis.top_finding.replace('_', ' ')}
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', marginLeft: 5 }}>
                  {(p.analysis.top_finding_confidence * 100).toFixed(0)}%
                </span>
              </span>
            )}
          </>
        ) : (
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>No patient selected — upload a scan to begin</span>
        )}
      </div>

      {/* Actions */}
      <div className="topbar-actions">
        {canReview && (
          <input
            className="notes-input"
            placeholder="Notes (optional)…"
            value={reviewNotes}
            onChange={e => setReviewNotes(e.target.value)}
          />
        )}
        <button className="btn btn-approve" disabled={!canReview} onClick={onApprove} title="Approve report">
          ✓ Approve
        </button>
        <button className="btn btn-revise" disabled={!canReview} onClick={onRevise} title="Request revision">
          ↺ Revise
        </button>
        <button className="btn btn-reject" disabled={!canReview} onClick={onReject} title="Reject report">
          ✗ Reject
        </button>
        <div className="topbar-sep" />
        <button className="btn btn-upload" disabled={isUploading} onClick={() => fileRef.current?.click()}>
          {isUploading ? '⏳ Uploading…' : '↑ Upload X-Ray'}
        </button>
        <input ref={fileRef} type="file" accept="image/*,.dcm" style={{ display: 'none' }} onChange={handleFile} />
      </div>
    </header>
  )
}
