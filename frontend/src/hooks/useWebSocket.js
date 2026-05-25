import { useEffect, useRef, useState } from 'react'
import { WS_BASE } from '../constants.js'

function classify(text) {
  if (text.includes('[Error]'))            return 't-error'
  if (text.includes('[Queue]'))            return 't-queue'
  if (text.includes('[Orchestrator]'))     return 't-orchestrator'
  if (text.includes('[Radiologist]'))      return 't-radiologist'
  if (text.includes('[VLM Review]'))       return 't-vlm'
  if (text.includes('[Clinical Advisor]')) return 't-advisor'
  if (text.includes('[Report Generator]')) return 't-report'
  if (text.includes('[Report]'))           return 't-report'
  if (text.includes('[HITL]'))             return 't-hitl'
  if (text.includes('[System]'))           return 't-system'
  return 't-default'
}

function mkEntry(text, type) {
  return {
    id:   Date.now() + Math.random(),
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    text,
    type,
  }
}

export function useWebSocket() {
  const [logs, setLogs] = useState([
    mkEntry('[System] Connecting to agent stream...', 't-system'),
  ])
  const wsRef   = useRef(null)
  const mounted = useRef(true)

  useEffect(() => {
    mounted.current = true

    const connect = () => {
      if (!mounted.current) return

      const ws = new WebSocket(`${WS_BASE}/ws/logs`)
      wsRef.current = ws

      ws.onopen = () => {
        if (!mounted.current) return
        setLogs(prev => [...prev, mkEntry('[System] Connected to agent log stream', 't-system')])
      }

      ws.onmessage = ({ data }) => {
        if (!mounted.current) return
        setLogs(prev => [...prev.slice(-300), mkEntry(data, classify(data))])
      }

      ws.onclose = () => {
        if (!mounted.current) return
        setLogs(prev => [...prev, mkEntry('[System] Connection lost — retrying in 3 s', 't-warning')])
        setTimeout(connect, 3000)
      }

      ws.onerror = () => ws.close()
    }

    connect()
    return () => {
      mounted.current = false
      wsRef.current?.close()
    }
  }, [])

  return logs
}
