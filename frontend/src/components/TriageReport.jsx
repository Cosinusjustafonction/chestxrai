import { API_BASE } from '../constants.js'

/* ── helpers ──────────────────────────────────────────────── */
function fmt(prob) { return (prob * 100).toFixed(1) + '%' }
function cls(prob) { return prob > 0.30 ? 'hi' : 'mid' }

function buildImpression(patient) {
  const { analysis, filename } = patient
  const { severity, top_finding, detected_pathologies } = analysis
  const stem     = filename.replace(/\.[^.]+$/, '')
  const detected = detected_pathologies ?? []

  let t = `Chest radiograph of ${stem} demonstrates `
  t += severity === 'CRITICAL' ? 'critical findings requiring immediate clinical correlation. '
     : severity === 'ABNORMAL' ? 'abnormal findings warranting further evaluation. '
     : 'no significant acute cardiopulmonary findings. '
  if (detected.length > 0) {
    t += `Primary finding: ${detected[0].pathology.replace('_', ' ')} (${fmt(detected[0].probability)} model confidence). `
    if (detected.length > 1)
      t += `Additional findings: ${detected.slice(1).map(p => p.pathology.replace('_', ' ')).join(', ')}. `
  }
  t += 'AI-assisted analysis performed using DenseNet121 trained on NIH ChestX-ray14 (112 K images, 14 classes). Clinical correlation required.'
  return t
}

function buildRecs(analysis, guidelines) {
  const recs = []
  if (analysis.severity === 'CRITICAL') {
    recs.push('Immediate radiologist review required — critical severity classification.')
    recs.push('Urgent clinical team notification recommended.')
  } else if (analysis.severity === 'ABNORMAL') {
    recs.push('Timely radiologist review recommended within clinical workflow.')
  } else {
    recs.push('Routine follow-up as clinically indicated.')
  }
  ;(guidelines ?? [])
    .filter(g => g?.found)
    .flatMap(g => g.follow_up?.slice(0, 1) ?? [])
    .forEach(f => recs.push(f))
  recs.push('Correlate with clinical history, physical examination, and prior imaging.')
  recs.push('Grad-CAM heatmaps available in X-Ray Viewer for explainability review.')
  return recs.slice(0, 6)
}

/* ── sub-components ───────────────────────────────────────── */
function PaperSection({ title, children }) {
  return (
    <div className="paper-section">
      <div className="paper-section-title">{title}</div>
      {children}
    </div>
  )
}

function GuidelineBlock({ g }) {
  if (!g?.found) return null
  const urgCls = g.urgency === 'emergent' ? 'emergent' : g.urgency === 'urgent' ? 'urgent' : 'routine'
  return (
    <div className="paper-guideline">
      <div className="pg-header">
        <span className="pg-name">{g.pathology?.replace('_', ' ')}</span>
        {g.urgency && <span className={`pg-urgency ${urgCls}`}>{g.urgency}</span>}
      </div>
      {g.definition && <p className="pg-def">{g.definition}</p>}
      {g.follow_up?.length > 0 && (
        <ul className="pg-fups">
          {g.follow_up.slice(0, 3).map((f, i) => <li key={i}>{f}</li>)}
        </ul>
      )}
    </div>
  )
}

function GradcamDownloadLinks({ patient }) {
  const paths = patient?.analysis?.gradcam_heatmap_paths ?? []
  if (paths.length === 0) return null
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 4 }}>
      {paths.map((p, i) => {
        const name   = p.split(/[\\/]/).pop()
        const match  = name.match(/_([^_]+)\.png$/i)
        const label  = match ? match[1] : `Heatmap ${i + 1}`
        const url    = `${API_BASE}/gradcam/${name}`
        return (
          <a
            key={i}
            href={url}
            download={`gradcam_${label}_${patient.id}.png`}
            target="_blank"
            rel="noreferrer"
            style={{
              fontSize: 11,
              fontFamily: 'Courier New, monospace',
              color: '#2563eb',
              textDecoration: 'underline',
              cursor: 'pointer',
              padding: '2px 0',
            }}
          >
            ↓ {label}
          </a>
        )
      })}
    </div>
  )
}

/* ── Paper document ───────────────────────────────────────── */
function PaperDocument({ patient }) {
  const { id, timestamp, status, analysis, guidelines, review, filename } = patient

  const detected   = analysis.detected_pathologies   ?? []
  const borderline = analysis.borderline_pathologies  ?? []
  const dateStr    = new Date(timestamp).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
  const timeStr    = new Date(timestamp).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
  const impression = buildImpression(patient)
  const recs       = buildRecs(analysis, guidelines)
  const guideFound = (guidelines ?? []).filter(g => g?.found)

  const stampCls = review?.decision === 'approve' ? 'approved'
                 : review?.decision === 'revise'  ? 'revise'
                 : review                         ? 'rejected'
                 : null

  return (
    <div className="paper">
      {/* Watermark stamp */}
      {stampCls && (
        <div className={`paper-stamp-overlay ${stampCls}`}>
          {review.decision === 'approve' ? 'APPROVED'
            : review.decision === 'revise' ? 'REVISION REQUESTED'
            : 'REJECTED'}
        </div>
      )}

      {/* Letterhead */}
      <div className="paper-letterhead">
        <div className="paper-facility">
          <div className="paper-facility-name">🫁 ChestXRAI Radiology</div>
          <div className="paper-facility-dept">Department of Diagnostic Radiology · AI-Assisted Triage Unit</div>
        </div>
        <div className="paper-report-type">
          <strong>Radiology Triage Report</strong>
          Report No: {id.toUpperCase()}<br />
          {dateStr} · {timeStr}
        </div>
      </div>

      {/* Patient info */}
      <div className="paper-info-grid">
        <div className="paper-info-row">
          <span className="info-lbl">Patient ID</span>
          <span className="info-val">#{id}</span>
        </div>
        <div className="paper-info-row">
          <span className="info-lbl">Date</span>
          <span className="info-val">{dateStr}</span>
        </div>
        <div className="paper-info-row">
          <span className="info-lbl">Source File</span>
          <span className="info-val" style={{ fontSize: 11 }}>{filename}</span>
        </div>
        <div className="paper-info-row">
          <span className="info-lbl">Modality</span>
          <span className="info-val">Chest X-Ray (CXR)</span>
        </div>
        <div className="paper-info-row">
          <span className="info-lbl">AI Model</span>
          <span className="info-val">DenseNet121 · NIH ChestX-ray14</span>
        </div>
        <div className="paper-info-row">
          <span className="info-lbl">Status</span>
          <span className="info-val" style={{ textTransform: 'uppercase', fontWeight: 700 }}>
            {status.replace('_', ' ')}
          </span>
        </div>
      </div>

      {/* Urgency */}
      <PaperSection title="Urgency Classification">
        <div className="paper-urgency">
          <span className={`urgency-stamp ${analysis.severity}`}>{analysis.severity}</span>
          <div>
            <div className="urgency-primary">
              {analysis.top_finding?.replace('_', ' ')}
            </div>
            <div className="urgency-conf-text">
              Model confidence: {fmt(analysis.top_finding_confidence)} &nbsp;·&nbsp;
              {detected.length} finding{detected.length !== 1 ? 's' : ''} detected
            </div>
          </div>
        </div>
      </PaperSection>

      {/* Detected findings */}
      {detected.length > 0 && (
        <PaperSection title={`Findings — Detected Pathologies (≥ 30% confidence)`}>
          <table className="paper-findings-table">
            <thead>
              <tr>
                <th>Pathology</th>
                <th>Probability</th>
                <th className="pf-bar-cell">Score</th>
                <th>Class</th>
              </tr>
            </thead>
            <tbody>
              {detected.map(p => (
                <tr key={p.pathology}>
                  <td className="pf-name">{p.pathology.replace('_', ' ')}</td>
                  <td><span className={`pf-prob ${cls(p.probability)}`}>{fmt(p.probability)}</span></td>
                  <td>
                    <div className="pf-bar-track">
                      <div className={`pf-bar-fill ${cls(p.probability)}`} style={{ width: fmt(p.probability) }} />
                    </div>
                  </td>
                  <td style={{ fontSize: 11, color: '#6b7280' }}>DETECTED</td>
                </tr>
              ))}
            </tbody>
          </table>
        </PaperSection>
      )}

      {/* Borderline */}
      {borderline.length > 0 && (
        <PaperSection title="Borderline Observations (15–30% confidence — monitor)">
          <table className="paper-findings-table">
            <thead>
              <tr>
                <th>Pathology</th>
                <th>Probability</th>
                <th className="pf-bar-cell">Score</th>
                <th>Class</th>
              </tr>
            </thead>
            <tbody>
              {borderline.map(p => (
                <tr key={p.pathology}>
                  <td className="pf-name">{p.pathology.replace('_', ' ')}</td>
                  <td><span className="pf-prob mid">{fmt(p.probability)}</span></td>
                  <td>
                    <div className="pf-bar-track">
                      <div className="pf-bar-fill mid" style={{ width: fmt(p.probability) }} />
                    </div>
                  </td>
                  <td style={{ fontSize: 11, color: '#6b7280' }}>BORDERLINE</td>
                </tr>
              ))}
            </tbody>
          </table>
        </PaperSection>
      )}

      {/* Guidelines */}
      {guideFound.length > 0 && (
        <PaperSection title="Clinical Guidelines">
          {guideFound.map((g, i) => <GuidelineBlock key={i} g={g} />)}
        </PaperSection>
      )}

      {/* Grad-CAM downloads */}
      <PaperSection title="Explainability — Grad-CAM Heatmaps">
        <p style={{ fontSize: 11.5, color: '#6b7280', marginBottom: 6 }}>
          Gradient-weighted Class Activation Mapping highlights regions driving each pathology prediction.
          Download individual heatmaps below:
        </p>
        <GradcamDownloadLinks patient={patient} />
      </PaperSection>

      {/* Impression */}
      <PaperSection title="Impression">
        <p className="paper-impression">{impression}</p>
      </PaperSection>

      {/* Recommendations */}
      <PaperSection title="Recommendations">
        <ul className="paper-recs">
          {recs.map((r, i) => (
            <li key={i}>
              <span className="prec-num">{String(i + 1).padStart(2, '0')}.</span>
              <span>{r}</span>
            </li>
          ))}
        </ul>
      </PaperSection>

      {/* AI Narrative from crew */}
      {patient.report && (
        <PaperSection title="AI Narrative Report">
          <p style={{ fontSize: 10.5, color: '#6b7280', marginBottom: 8, fontStyle: 'italic' }}>
            Raw narrative synthesised by the multi-agent crew (Radiologist → VLM Review → Clinical Advisor → Report Generator).
          </p>
          <pre className="paper-narrative">{patient.report}</pre>
        </PaperSection>
      )}

      {/* Review result */}
      {review && (
        <PaperSection title="Physician Review">
          <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start', flexWrap: 'wrap' }}>
            <span className={`urgency-stamp ${review.decision === 'approve' ? 'NORMAL' : 'CRITICAL'}`}
              style={{ fontSize: 11, padding: '3px 12px' }}>
              {review.decision === 'approve' ? '✓ APPROVED'
                : review.decision === 'revise' ? '↺ REVISION REQUESTED'
                : '✗ REJECTED'}
            </span>
            {review.notes && (
              <span style={{ fontSize: 12, color: '#374151', fontStyle: 'italic' }}>
                "{review.notes}"
              </span>
            )}
            <span style={{ fontSize: 11, fontFamily: 'Courier New, monospace', color: '#9ca3af', marginLeft: 'auto' }}>
              {new Date(review.timestamp).toLocaleString()}
            </span>
          </div>
        </PaperSection>
      )}

      {/* Footer */}
      <div className="paper-footer">
        <div className="paper-sig-block">
          <div className="paper-sig-line" />
          <div className="paper-sig-label">Radiologist Signature</div>
        </div>
        <div className="paper-sig-block" style={{ alignItems: 'center' }}>
          <div className="paper-sig-line" />
          <div className="paper-sig-label">Date / Stamp</div>
        </div>
        <div className="paper-ai-note">
          Generated by ChestXRAI AI Triage System.<br />
          Not a substitute for clinical judgement.
        </div>
      </div>
    </div>
  )
}

/* ── Main export ──────────────────────────────────────────── */
export default function TriageReport({ patient }) {
  const handlePrint = () => window.print()

  const handleDownloadAllGradcam = () => {
    const paths = patient?.analysis?.gradcam_heatmap_paths ?? []
    paths.forEach((p, i) => {
      setTimeout(() => {
        const name  = p.split(/[\\/]/).pop()
        const match = name.match(/_([^_]+)\.png$/i)
        const label = match ? match[1] : `heatmap_${i}`
        const a     = document.createElement('a')
        a.href      = `${API_BASE}/gradcam/${name}`
        a.download  = `gradcam_${label}_${patient.id}.png`
        a.target    = '_blank'
        a.click()
      }, i * 300)
    })
  }

  const hasGradcam = (patient?.analysis?.gradcam_heatmap_paths ?? []).length > 0

  return (
    <div className="triage-report">
      {/* Toolbar */}
      <div className="paper-toolbar">
        <span className="paper-toolbar-title">Triage Report</span>
        <div className="paper-toolbar-actions">
          {hasGradcam && (
            <button className="btn-paper dl" onClick={handleDownloadAllGradcam}>
              ↓ All Grad-CAMs
            </button>
          )}
          <button className="btn-paper" onClick={handlePrint} disabled={!patient?.analysis}>
            🖨 Print / PDF
          </button>
        </div>
      </div>

      {/* Scrollable paper area */}
      <div className="paper-scroll">
        {/* No patient */}
        {!patient && (
          <div className="paper-state">
            <div className="paper-state-icon">📋</div>
            <div className="paper-state-title">No patient selected.</div>
            <div className="paper-state-sub">Select a patient from the queue or upload a chest X-ray to generate a report.</div>
          </div>
        )}

        {/* Pending / analyzing */}
        {patient && !patient.analysis && patient.status !== 'error' && (
          <div className="paper-state">
            <div className="paper-state-icon" style={{ animation: 'spin-slow 2s linear infinite', display: 'inline-block' }}>⚙️</div>
            <div className="paper-state-title">
              {patient.status === 'queued' ? 'Queued for analysis…' : 'AI analysis in progress…'}
            </div>
            <div className="paper-state-sub">
              DenseNet121 is processing the chest radiograph.<br />
              The report will appear here once complete.
            </div>
          </div>
        )}

        {/* Error */}
        {patient && patient.status === 'error' && (
          <div className="paper-state">
            <div className="paper-state-icon" style={{ opacity: .4 }}>⚠️</div>
            <div className="paper-state-title" style={{ color: '#dc2626' }}>Analysis failed</div>
            <div className="paper-state-sub">{patient.error ?? 'Check the Agent Activity log for details.'}</div>
          </div>
        )}

        {/* Full paper report */}
        {patient?.analysis && <PaperDocument patient={patient} />}
      </div>
    </div>
  )
}
