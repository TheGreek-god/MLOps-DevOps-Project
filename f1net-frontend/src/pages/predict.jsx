import { useState } from 'react'
import axios from 'axios'
import Podium from '../components/Podium'
import ResultsTable from '../components/ResultsTable'

const API_BASE = import.meta.env.VITE_PREDICT_API_URL

const teamColors = {
  'Mercedes': '#00D2BE',
  'Red Bull Racing': '#0600EF',
  'Ferrari': '#DC0000',
  'McLaren': '#FF8700',
  'Aston Martin': '#229971',
  'Alpine': '#0090FF',
  'Williams': '#005AFF',
  'Haas': '#B6BABD',
  'Kick Sauber': '#52E252',
  'RB': '#6692FF',
  'Cadillac': '#C8A951',
}

export default function Predict() {
  const [year, setYear] = useState(2026)
  const [round, setRound] = useState(1)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handlePredict() {
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const res = await axios.post(API_BASE + '/predict', { year: parseInt(year), round_num: parseInt(round) })
      setResults(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to fetch predictions')
    } finally {
      setLoading(false)
    }
  }

  const podium = results?.predictions?.slice(0, 3) || []
  const rest = results?.predictions?.slice(3) || []

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
          Race <span style={{ color: 'var(--red)' }}>Predictor</span>
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
          ML-powered race outcome predictions using qualifying and practice data
        </p>
      </div>

      <div style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: '12px',
        padding: '1.5rem',
        marginBottom: '2rem',
        display: 'flex',
        alignItems: 'flex-end',
        gap: '1rem',
        flexWrap: 'wrap',
      }}>
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
        <button
          onClick={handlePredict}
          disabled={loading}
          style={{
            background: loading ? 'var(--red-dim)' : 'var(--red)',
            color: '#fff',
            padding: '10px 28px',
            borderRadius: '8px',
            fontFamily: 'var(--font-display)',
            fontSize: '16px',
            fontWeight: 700,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            transition: 'background 0.15s',
            height: '42px',
          }}
        >
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </div>

      {error && (
        <div style={{
          background: '#1a0a0a',
          border: '1px solid #3a1010',
          borderRadius: '8px',
          padding: '1rem 1.25rem',
          color: '#ff6b6b',
          fontSize: '14px',
          marginBottom: '2rem',
        }}>
          {error}
        </div>
      )}

      {results && (
        <>
          <div style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '12px',
              color: 'var(--text-muted)',
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
            }}>
              {year} · Round {results.race_idx} · {results.total_cars} cars
            </span>
            {!results.ground_truth_available && (
              <span style={{
                background: '#1a1000',
                border: '1px solid #3a2800',
                borderRadius: '4px',
                padding: '2px 8px',
                fontSize: '11px',
                color: '#f59e0b',
                fontFamily: 'var(--font-mono)',
              }}>
                Pre-race prediction
              </span>
            )}
          </div>
          <Podium podium={podium} teamColors={teamColors} />
          <ResultsTable results={rest} teamColors={teamColors} startPos={4} />
        </>
      )}
    </div>
  )
}