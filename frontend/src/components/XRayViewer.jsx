import { useState, useRef, useCallback, useEffect } from 'react'
import { API_BASE, PATHOLOGIES } from '../constants.js'

function srcFromPath(filepath) {
  if (!filepath) return null
  return `${API_BASE}/uploads/${filepath.split(/[\\/]/).pop()}`
}
function gradcamFromPath(abspath) {
  if (!abspath) return null
  return `${API_BASE}/gradcam/${abspath.split(/[\\/]/).pop()}`
}
function barClass(p) { return p > 0.30 ? 'hi' : p > 0.15 ? 'mid' : 'lo' }

/* ── Zoom / Pan image panel ───────────────────────────────── */
function ZoomPanPanel({ src, label, fallback, downloadName }) {
  const [scale,      setScale]      = useState(1)
  const [offset,     setOffset]     = useState({ x: 0, y: 0 })
  const [dragging,   setDragging]   = useState(false)
  const [imgFailed,  setImgFailed]  = useState(false)
  const dragOrigin = useRef(null)

  // Reset when src changes
  useEffect(() => { setScale(1); setOffset({ x: 0, y: 0 }); setImgFailed(false) }, [src])

  const onWheel = useCallback((e) => {
    e.preventDefault()
    const factor = e.deltaY < 0 ? 1.12 : 0.9
    setScale(s => Math.min(Math.max(s * factor, 0.5), 12))
  }, [])

  const onMouseDown = (e) => {
    if (e.button !== 0) return
    setDragging(true)
    dragOrigin.current = { x: e.clientX - offset.x, y: e.clientY - offset.y }
  }
  const onMouseMove = (e) => {
    if (!dragging || !dragOrigin.current) return
    setOffset({ x: e.clientX - dragOrigin.current.x, y: e.clientY - dragOrigin.current.y })
  }
  const onMouseUp   = () => setDragging(false)
  const reset       = () => { setScale(1); setOffset({ x: 0, y: 0 }) }

  const zoomIn  = (e) => { e.stopPropagation(); setScale(s => Math.min(s * 1.3, 12)) }
  const zoomOut = (e) => { e.stopPropagation(); setScale(s => Math.max(s * 0.77, 0.5)) }

  const handleDownload = (e) => {
    e.stopPropagation()
    if (!src) return
    const a = document.createElement('a')
    a.href = src
    a.download = downloadName || label.replace(/\s+/g, '_') + '.png'
    a.target = '_blank'
    a.click()
  }

  return (
    <div
      className={`img-panel ${dragging ? 'dragging' : ''}`}
      onWheel={onWheel}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
    >
      <span className="img-panel-tag">{label}</span>

      {src && !imgFailed ? (
        <img
          src={src}
          alt={label}
          draggable={false}
          onError={() => setImgFailed(true)}
          style={{
            transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
            transition: dragging ? 'none' : 'transform 0.08s ease',
          }}
        />
      ) : (
        <div className="img-placeholder">
          <div className="img-placeholder-icon">🩻</div>
          <div className="img-placeholder-text">{fallback}</div>
        </div>
      )}

      {/* Download button (GradCAM panels only) */}
      {src && !imgFailed && downloadName && (
        <button className="img-download-btn" onClick={handleDownload} title="Download heatmap">
          ↓ Download
        </button>
      )}

      {/* Zoom controls */}
      <div className="zoom-controls" onMouseDown={e => e.stopPropagation()}>
        <button className="zoom-btn" onClick={zoomIn}  title="Zoom in">+</button>
        <button className="zoom-btn pct" onClick={reset} title="Reset zoom">
          {Math.round(scale * 100)}%
        </button>
        <button className="zoom-btn" onClick={zoomOut} title="Zoom out">−</button>
      </div>
    </div>
  )
}

/* ── Main component ───────────────────────────────────────── */
export default function XRayViewer({ patient, isLoading }) {
  const [hmIdx, setHmIdx] = useState(0)

  const analysis  = patient?.analysis
  const hmPaths   = analysis?.gradcam_heatmap_paths ?? []
  const allPreds  = analysis?.all_predictions ?? {}

  useEffect(() => setHmIdx(0), [patient?.id])

  const srcUrl = srcFromPath(patient?.filepath)
  const hmUrl  = hmPaths[hmIdx] ? gradcamFromPath(hmPaths[hmIdx]) : null

  const hmOptions = hmPaths.map((p, i) => {
    const m = p.match(/_([^_/\\]+)\.png$/i)
    return { label: m ? m[1] : `Heatmap ${i + 1}`, value: i }
  })

  const hmDownloadName = hmOptions[hmIdx]
    ? `gradcam_${hmOptions[hmIdx].label}_${patient?.id}.png`
    : undefined

  const chartData = PATHOLOGIES
    .map(p => ({ name: p, val: allPreds[p] ?? 0, cls: barClass(allPreds[p] ?? 0) }))
    .sort((a, b) => b.val - a.val)

  return (
    <section className="xray-viewer">
      {/* Header */}
      <div className="panel-header">
        <span className="panel-title">X-Ray Viewer — scroll to zoom · drag to pan</span>
        <div className="viewer-controls">
          {hmOptions.length > 0 && (
            <>
              <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                Grad-CAM
              </span>
              <select
                className="heatmap-select"
                value={hmIdx}
                onChange={e => setHmIdx(Number(e.target.value))}
              >
                {hmOptions.map(o => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </>
          )}
        </div>
      </div>

      {/* Side-by-side panels */}
      <div className="image-row">
        <ZoomPanPanel
          src={srcUrl}
          label="Original X-Ray"
          fallback={patient ? 'Mount /uploads in FastAPI to display' : 'No scan selected'}
        />
        <ZoomPanPanel
          src={hmUrl}
          label={`Grad-CAM — ${hmOptions[hmIdx]?.label ?? '—'}`}
          fallback={
            !patient        ? 'No scan selected'         :
            !analysis       ? 'Awaiting AI analysis…'    :
            hmPaths.length === 0 ? 'No heatmaps generated'  :
            'Mount /gradcam in FastAPI to display'
          }
          downloadName={hmDownloadName}
        />
      </div>

      {/* Pathology bar chart */}
      {chartData.some(d => d.val > 0) && (
        <div className="path-chart">
          <div className="chart-title">Pathology Detection Scores — DenseNet121 (14 classes)</div>
          <div className="chart-grid">
            {chartData.map(({ name, val, cls }) => (
              <div key={name} className="chart-row">
                <span className="chart-lbl">{name.replace('_', ' ')}</span>
                <div className="chart-track">
                  <div className={`chart-fill ${cls}`} style={{ width: `${(val * 100).toFixed(1)}%` }} />
                </div>
                <span className={`chart-val ${cls}`}>{(val * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {isLoading && <div className="load-overlay"><div className="spinner" /></div>}
    </section>
  )
}
