import { useState } from 'react'
import { usePatients }   from './hooks/usePatients.js'
import { useWebSocket }  from './hooks/useWebSocket.js'
import TopBar            from './components/TopBar.jsx'
import PatientQueue      from './components/PatientQueue.jsx'
import XRayViewer        from './components/XRayViewer.jsx'
import TriageReport      from './components/TriageReport.jsx'
import AgentLog          from './components/AgentLog.jsx'
import RunLogs           from './components/RunLogs.jsx'

export default function App() {
  const {
    patients, selectedId, setSelectedId,
    selectedPatient, isLoading,
    uploadXray, reviewPatient, stats,
  } = usePatients()

  const logs = useWebSocket()

  const [isUploading, setIsUploading] = useState(false)
  const [reviewNotes, setReviewNotes] = useState('')
  const [showLogs, setShowLogs]       = useState(false)

  const handleUpload = async (file) => {
    setIsUploading(true)
    try { await uploadXray(file) }
    catch (e) { console.error('Upload error:', e) }
    finally   { setIsUploading(false) }
  }

  const handleReview = async (decision) => {
    if (!selectedId) return
    await reviewPatient(selectedId, decision, reviewNotes)
    setReviewNotes('')
  }

  return (
    <div className="app-shell">
      <TopBar
        onUpload={handleUpload}
        onApprove={() => handleReview('approve')}
        onReject={() => handleReview('reject')}
        onRevise={() => handleReview('revise')}
        canReview={selectedPatient?.status === 'awaiting_review'}
        isUploading={isUploading}
        selectedPatient={selectedPatient}
        reviewNotes={reviewNotes}
        setReviewNotes={setReviewNotes}
      />

      <div className="app-body">
        <PatientQueue
          patients={patients}
          selectedId={selectedId}
          onSelect={setSelectedId}
          stats={stats}
        />

        <div className="center-pane">
          <XRayViewer patient={selectedPatient} isLoading={isLoading} />
          <TriageReport patient={selectedPatient} />
        </div>

        <AgentLog logs={logs} onOpenLogs={() => setShowLogs(true)} />
      </div>

      {showLogs && (
        <RunLogs patients={patients} onClose={() => setShowLogs(false)} />
      )}
    </div>
  )
}
