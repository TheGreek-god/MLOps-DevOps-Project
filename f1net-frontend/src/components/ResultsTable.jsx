export default function ResultsTable({ results, teamColors, startPos }) {
  if (!results.length) return null

  return (
    <div style={{ marginBottom: '2rem' }}>
      <p style={{
        fontFamily: 'var(--font-mono)',
        fontSize: '11px',
        color: 'var(--text-muted)',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        marginBottom: '1rem',
      }}>Full Classification</p>

      <div style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: '12px',
        overflow: 'hidden',
      }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Pos', 'Driver', 'Team', 'Score', 'Actual'].map(h => (
                <th key={h} style={{
                  padding: '10px 16px',
                  textAlign: h === 'Pos' ? 'center' : 'left',
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
            {results.map((item, i) => {
              const teamColor = teamColors[item.team] || '#6b6b6b'
              const pos = startPos + i
              return (
                <tr key={i} style={{
                  borderBottom: i < results.length - 1 ? '1px solid var(--border)' : 'none',
                  transition: 'background 0.1s',
                }}>
                  <td style={{
                    padding: '12px 16px',
                    textAlign: 'center',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '13px',
                    color: 'var(--text-muted)',
                    width: '60px',
                  }}>
                    P{pos}
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
                      <span style={{
                        fontFamily: 'var(--font-display)',
                        fontSize: '16px',
                        fontWeight: 600,
                        letterSpacing: '0.03em',
                        textTransform: 'uppercase',
                      }}>
                        {item.driver}
                      </span>
                    </div>
                  </td>
                  <td style={{
                    padding: '12px 16px',
                    fontSize: '13px',
                    color: 'var(--text-dim)',
                  }}>
                    {item.team || '—'}
                  </td>
                  <td style={{
                    padding: '12px 16px',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '13px',
                    color: 'var(--text-muted)',
                  }}>
                    {item.model_raw_score?.toFixed(3)}
                  </td>
                  <td style={{ padding: '12px 16px' }}>
                    {item.actual_p ? (
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '13px',
                        color: item.exact_match_hit ? 'var(--green)' : 'var(--text-muted)',
                      }}>
                        {item.exact_match_hit ? '✓ ' : ''}P{item.actual_p}
                      </span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>—</span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}