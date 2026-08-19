import { useState } from 'react'
import axios from 'axios'

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

function getHeatColor(error, maxError) {
  const ratio = Math.min(error / maxError, 1)
  const hue = (1 - ratio) * 120
  return `hsla(${hue}, 70%, 35%,.25)`
}

function getHeatTextColor(error, maxError) {
  const ratio = Math.min(error / maxError, 1)
  const hue = (1 - ratio) * 120
  return `hsl(${hue}, 80%, 65%)`
}

export default function Compare() {
  const [year, setYear] = useState(2026)
  const [round, setRound] = useState(1)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleCompare() {
    setLoading(true)
    setError(null)
    setResults(null)
    try {
      const res = await axios.post(API_BASE + '/predict', { year: parseInt(year), round_num: parseInt(round) })
      if (!res.data.ground_truth_available) {
        setError('No race results available for this round yet. Run /ingest/finish first.')
        return
      }
      setResults(res.data)
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to fetch data')
    } finally {
      setLoading(false)
    }
  }

  const predictions = results?.predictions || []
  const totalCars = results?.total_cars || 20
  const exactHits = predictions.filter(p => p.exact_match_hit).length

  const sortedByActual = [...predictions].sort((a, b) => a.actual_p - b.actual_p)

  const predictedRanks = sortedByActual.map(d => d.predicted_p)
  const actualRanks = sortedByActual.map(d => d.actual_p)

  function spearman(x, y) {
    const n = x.length
    const rankX = x.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v).map((r, rank) => ({ ...r, rank: rank + 1 })).sort((a, b) => a.i - b.i).map(r => r.rank)
    const rankY = y.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v).map((r, rank) => ({ ...r, rank: rank + 1 })).sort((a, b) => a.i - b.i).map(r => r.rank)
    const dSq = rankX.reduce((sum, r, i) => sum + Math.pow(r - rankY[i], 2), 0)
    return 1 - (6 * dSq) / (n * (n * n - 1))
  }

  const rho = predictions.length > 1 ? spearman(predictedRanks, actualRanks) : null

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
          Race <span style={{ color: 'var(--red)' }}>Analysis</span>
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
          Compare predicted vs actual race results
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
          onClick={handleCompare}
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
          {loading ? 'Loading...' : 'Compare'}
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
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '12px',
            marginBottom: '2rem',
          }}>
            {[
              { label: 'Spearman ρ', value: rho ? rho.toFixed(3) : '—', color: rho > 0.6 ? 'var(--green)' : 'var(--text)' },
              { label: 'Exact hits', value: `${exactHits} / ${totalCars}`, color: 'var(--text)' },
              { label: 'Round', value: `${year} · R${results.race_idx}`, color: 'var(--text)' },
            ].map(stat => (
              <div key={stat.label} style={{
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderRadius: '10px',
                padding: '1rem 1.25rem',
              }}>
                <p style={{ fontSize: '11px', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '6px' }}>
                  {stat.label}
                </p>
                <p style={{ fontFamily: 'var(--font-display)', fontSize: '28px', fontWeight: 700, color: stat.color, lineHeight: 1 }}>
                  {stat.value}
                </p>
              </div>
            ))}
          </div>

          <div style={{
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: '12px',
            overflow: 'hidden',
          }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Actual P', 'Driver', 'Predicted P', 'Error'].map(h => (
                    <th key={h} style={{
                      padding: '10px 16px',
                      textAlign: 'left',
                      fontFamily: 'var(--font-mono)',
                      fontSize: '11px',
                      color: 'var(--text-muted)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.1em',
                      fontWeight: 400,
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedByActual.map((item, i) => {
                  const err = Math.abs(item.predicted_p - item.actual_p)
                  const teamColor = teamColors[item.team] || '#6b6b6b'
                  return (
                    <tr key={i} style={{
                        borderBottom: i < sortedByActual.length - 1 ? '1px solid var(--border)' : 'none',
                        background: 'transparent',
                        boxShadow: err === 0 
  ? 'inset 0 0 12px 2px rgba(34,197,94,0.25), inset 0 0 0 1px rgba(34,197,94,0.5)' 
  : `inset 0 0 8px 1px hsla(${(1 - Math.min(err / (totalCars-1), 1)) * 120}, 70%, 45%, 0.15), inset 0 0 0 1px hsla(${(1 - Math.min(err / (totalCars-1), 1)) * 120}, 70%, 45%, 0.2)`,
                        }}>
                        <td style={{
                        padding: '12px 16px',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '14px',
                        fontWeight: 500,
                        color: err === 0 ? 'var(--green)' : getHeatTextColor(err, totalCars - 1),
                        width: '80px',
                      }}>
                        P{item.actual_p}
                      </td>
                      <td style={{ padding: '12px 16px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <div style={{
                            width: '3px',
                            height: '32px',
                            background: teamColor,
                            borderRadius: '2px',
                            flexShrink: 0,
                          }} />
                          <div>
                            <p style={{
                              fontFamily: 'var(--font-display)',
                              fontSize: '16px',
                              fontWeight: 600,
                              textTransform: 'uppercase',
                              letterSpacing: '0.03em',
                            }}>
                              {item.driver}
                            </p>
                            <p style={{ fontSize: '12px', color: err === 0 ? 'var(--green)' : getHeatTextColor(err, totalCars - 1), }}>
                              {item.team || ''}
                            </p>
                          </div>
                        </div>
                      </td>
                      <td style={{
                        padding: '12px 16px',
                        fontFamily: 'var(--font-mono)',
                        fontSize: '14px',
                        color:err === 0 ? 'var(--green)' : getHeatTextColor(err, totalCars - 1),
                      }}>
                        {err === 0 ? '✓ ' : ''}P{item.predicted_p}
                      </td>
                      <td style={{ padding: '12px 16px' }}>
                        <span style={{
                          display: 'inline-block',
                          padding: '3px 10px',
                          borderRadius: '4px',
                          background: err === 0 ? 'rgba(34,197,94,0.15)' : getHeatColor(err, totalCars - 1),
                          color: err === 0 ? 'var(--green)' : getHeatTextColor(err, totalCars - 1),
                          fontFamily: 'var(--font-mono)',
                          fontSize: '12px',
                          fontWeight: 500,
                        }}>
                          {err === 0 ? 'exact' : `±${err}`}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}