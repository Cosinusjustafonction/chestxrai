import { useState, useEffect, useCallback, useRef } from 'react'
import { API_BASE } from '../constants.js'

async function apiFetch(path, opts) {
  const res = await fetch(`${API_BASE}${path}`, opts)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export function usePatients() {
  const [patients,        setPatients]        = useState([])
  const [selectedId,      setSelectedId]      = useState(null)
  const [selectedPatient, setSelectedPatient] = useState(null)
  const [isLoading,       setIsLoading]       = useState(false)

  // Keep a ref so callbacks don't stale-close over selectedId
  const selIdRef = useRef(selectedId)
  selIdRef.current = selectedId

  /* ── queue polling ────────────────────────────────────────── */
  const fetchQueue = useCallback(async () => {
    try {
      const data = await apiFetch('/queue')
      setPatients(data)
    } catch {}
  }, [])

  useEffect(() => {
    fetchQueue()
    const iv = setInterval(fetchQueue, 3000)
    return () => clearInterval(iv)
  }, [fetchQueue])

  /* ── auto-select first patient if none chosen ─────────────── */
  useEffect(() => {
    if (!selIdRef.current && patients.length > 0) {
      setSelectedId(patients[0].id)
    }
  }, [patients])

  /* ── patient detail ───────────────────────────────────────── */
  const fetchPatient = useCallback(async (id) => {
    if (!id) { setSelectedPatient(null); return }
    setIsLoading(true)
    try {
      const data = await apiFetch(`/patient/${id}`)
      setSelectedPatient(data)
    } catch {} finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchPatient(selectedId)
  }, [selectedId, fetchPatient])

  /* ── fast-poll while analysis is running ──────────────────── */
  useEffect(() => {
    const status = selectedPatient?.status
    if (status !== 'queued' && status !== 'analyzing') return
    const iv = setInterval(() => fetchPatient(selectedId), 2000)
    return () => clearInterval(iv)
  }, [selectedId, selectedPatient?.status, fetchPatient])

  /* ── actions ──────────────────────────────────────────────── */
  const uploadXray = useCallback(async (file) => {
    const body = new FormData()
    body.append('file', file)
    const data = await apiFetch('/upload', { method: 'POST', body })
    setSelectedId(data.patient_id)
    await fetchQueue()
    return data
  }, [fetchQueue])

  const reviewPatient = useCallback(async (id, decision, notes = '') => {
    await apiFetch(
      `/review/${id}?decision=${encodeURIComponent(decision)}&notes=${encodeURIComponent(notes)}`,
      { method: 'POST' }
    )
    await Promise.all([fetchQueue(), fetchPatient(id)])
  }, [fetchQueue, fetchPatient])

  /* ── derived stats ────────────────────────────────────────── */
  const stats = {
    total:   patients.length,
    pending: patients.filter(p => p.status === 'awaiting_review').length,
    done:    patients.filter(p => p.status === 'approved').length,
  }

  return {
    patients,
    selectedId,
    setSelectedId,
    selectedPatient,
    isLoading,
    uploadXray,
    reviewPatient,
    stats,
  }
}
