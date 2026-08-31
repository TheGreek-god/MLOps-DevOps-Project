import { useState } from 'react'
import axios from 'axios'

const INGEST_BASE = ''

export default function Admin() {
  const [year, setYear] = useState(2026)
  const [round, setRound] = useState(1)
  const [jobs, setJobs] = useState([])
  const [loading, setLoading] = useState(null)
  const [apiKey, setApiKey] = useState('')

  async function triggerJob(endpoint, label) {
    setLoading(endpoint)
    try {
      const res = await axios.post(`${INGEST_BASE}${endpoint}`, 
        { year: parseInt(year), round: parseInt(round) },
        { headers: { 'X-API-KEY': apiKey } }
      )
      const jobId = res.data.job_id
      setJobs(prev => [{
        id: jobId,
        label,
        year,
        round,
        status: 'queued',
        message: 'Job queued.',
        time: new Date().toLocaleTimeString()
      }, ...prev])
      pollStatus(jobId)
    } catch (e) {
      setJobs(prev => [{
        id: 'err-' + Date.now(),
        label,
        year,
        round,
        status: 'error',
        message: e.response?.data?.detail || 'Request failed',
        time: new Date().toLocaleTimeString()
      }, ...prev])
    } finally {
      setLoading(null)
    }
  }

  async function pollStatus(jobId) {
    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${INGEST_BASE}/ingest/status/${jobId}`, {
          headers: { 'X-API-KEY': apiKey }
        })
        const data = res.data
        setJobs(prev => prev.map(j => j.id === jobId ? { ...j, ...data } : j))
        if (data.status === 'done' || data.status === 'error') {
          clearInterval(interval)
        }
      } catch {
        clearInterval(interval)
      }
    }, 2000)
  }

  const statusColor = (status) => ({
    queued: '#f59e0b',
    running: '#3b82f6',
    done: '#22c55e',
    error: '#ef4444',
  }[status] || 'var(--text-muted)')

  return (
    
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '2rem' }}>
      <div style={{ marginBottom: '2.5rem' }}>
        <h1 style={{
          fontFamily: 'var(--font-display)',
          fontSize: '48px',
          fontWeight: 700,
          letterSpacing: '0.04em',
          textTransform: 'uppercase',
          lineHeight: 1,
          marginBottom: '0.5rem',
        }}>
          Admin <span style={{ color: 'var(--red)' }}>Panel</span>
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
          Ingest race data and trigger model updates
        </p>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '1.25rem' }}>
        <label style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
          API Key
        </label>
        <input
          type="password"
          value={apiKey}
          onChange={e => setApiKey(e.target.value)}
          placeholder="Enter admin API key"
          style={{
            background: 'var(--surface-2)',
            border: '1px solid var(--border)',
            borderRadius: '8px',
            color: 'var(--text)',
            padding: '10px 14px',
            fontSize: '15px',
            width: '300px',
            outline: 'none',
          }}
        />
      </div>

      <div style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: '12px',
        padding: '1.5rem',
        marginBottom: '2rem',
      }}>
        <p style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '11px',
          color: 'var(--text-muted)',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
          marginBottom: '1.25rem',
        }}>Race Selection</p>

        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '1rem', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
              Season
            </label>
            <input
              type="number"
              value={year}
              onChange={e => setYear(e.target.value)}
              min={2026}
              style={{
                background: 'var(--surface-2)',
                border: '1px solid var(--border)',
                borderRadius: '8px',
                color: 'var(--text)',
                padding: '10px 14px',
                fontSize: '15px',
                width: '110px',
                outline: 'none',
              }}
            />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <label style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
              Round
            </label>
            <input
              type="number"
              value={round}
              onChange={e => setRound(e.target.value)}
              min={1}
              max={24}
              style={{
                background: 'var(--surface-2)',
                border: '1px solid var(--border)',
                borderRadius: '8px',
                color: 'var(--text)',
                padding: '10px 14px',
                fontSize: '15px',
                width: '110px',
                outline: 'none',
              }}
            />
          </div>
        </div>

        <div style={{ display: 'flex', gap: '1rem', marginTop: '1.25rem', flexWrap: 'wrap' }}>
          <button
            onClick={() => triggerJob('/ingest', 'Ingest')}
            disabled={loading !== null}
            style={{
              background: loading === '/ingest' ? 'var(--red-dim)' : 'var(--red)',
              color: '#fff',
              padding: '10px 24px',
              borderRadius: '8px',
              fontFamily: 'var(--font-display)',
              fontSize: '15px',
              fontWeight: 700,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              transition: 'background 0.15s',
              opacity: loading !== null && loading !== '/ingest' ? 0.5 : 1,
            }}
          >
            {loading === '/ingest' ? 'Queuing...' : 'Ingest Race Weekend'}
          </button>

          <button
            onClick={() => triggerJob('/ingest/finish', 'Ingest Results')}
            disabled={loading !== null}
            style={{
              background: 'var(--surface-2)',
              color: 'var(--text)',
              border: '1px solid var(--border)',
              padding: '10px 24px',
              borderRadius: '8px',
              fontFamily: 'var(--font-display)',
              fontSize: '15px',
              fontWeight: 700,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              transition: 'background 0.15s',
              opacity: loading !== null && loading !== '/ingest/finish' ? 0.5 : 1,
            }}
          >
            {loading === '/ingest/finish' ? 'Queuing...' : 'Ingest Results'}
          </button>
        </div>
      </div>

      {jobs.length > 0 && (
        <div>
          <p style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '11px',
            color: 'var(--text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
            marginBottom: '1rem',
          }}>Job Log</p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {jobs.map(job => (
              <div key={job.id} style={{
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: '10px',
                padding: '1rem 1.25rem',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{
                      width: '8px',
                      height: '8px',
                      borderRadius: '50%',
                      background: statusColor(job.status),
                      display: 'inline-block',
                      flexShrink: 0,
                    }} />
                    <span style={{
                      fontFamily: 'var(--font-display)',
                      fontSize: '15px',
                      fontWeight: 600,
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                    }}>
                      {job.label} · {job.year} R{job.round}
                    </span>
                  </div>
                  <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '11px',
                    color: 'var(--text-muted)',
                  }}>
                    {job.time}
                  </span>
                </div>
                <p style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '12px',
                  color: statusColor(job.status),
                  marginLeft: '18px',
                }}>
                  {job.status} — {job.message}
                </p>
                {job.rows_added && (
                  <p style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '11px',
                    color: 'var(--text-muted)',
                    marginLeft: '18px',
                    marginTop: '4px',
                  }}>
                    {job.rows_added} rows added · {job.new_drivers?.length > 0 ? `New drivers: ${job.new_drivers.join(', ')}` : 'No new drivers'}
                  </p>
                )}
                {job.finetune_job_id && (
                  <p style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '11px',
                    color: '#3b82f6',
                    marginLeft: '18px',
                    marginTop: '4px',
                  }}>
                    Finetune triggered · Job {job.finetune_job_id.slice(0, 8)}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}