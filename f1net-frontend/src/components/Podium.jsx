   import ant23 from '../assets/drivers/ka12.jpg'
  import rus63 from '../assets/drivers/gr63.jpg'
  import nor4 from '../assets/drivers/ln1.jpg'
  import pia81 from '../assets/drivers/op81.jpg'
  import ver1 from '../assets/drivers/mv3.jpg'
  import lec16 from '../assets/drivers/cl16.jpg'
  import ham44 from '../assets/drivers/lh44.jpg'
  import alb23 from '../assets/drivers/aa23.jpg'
  import alo14 from '../assets/drivers/fa14.jpg'
  import gas10 from '../assets/drivers/pg10.jpg'
  import hul27 from '../assets/drivers/nh27.jpg'
  import law30 from '../assets/drivers/ll30.jpg'
  import ocn31 from '../assets/drivers/eo31.jpg'
  import str18 from '../assets/drivers/ls18.jpg'
  import bor5 from '../assets/drivers/gb5.jpg'
  import bea87 from '../assets/drivers/ob87.jpg'
  import had6 from '../assets/drivers/ih6.jpg'
  import lin45 from '../assets/drivers/al41.jpg'
  import col43 from '../assets/drivers/fc43.jpg'
  import sai55 from '../assets/drivers/cs55.jpg'
export default function Podium({ podium, teamColors }) {
  if (!podium.length) return null

  const order = [1, 0, 2]
  const heights = [140, 180, 110]
  const labels = ['P2', 'P1', 'P3']

 

  const driverPhotos = {
  'Kimi Antonelli': ant23,
  'George Russell': rus63,
  'Lando Norris': nor4,
  'Oscar Piastri': pia81,
  'Max Verstappen': ver1,
  'Charles Leclerc': lec16,
  'Lewis Hamilton': ham44,
  'Alexander Albon': alb23,
  'Fernando Alonso': alo14,
  'Pierre Gasly': gas10,
  'Nico Hulkenberg': hul27,
  'Liam Lawson': law30,
  'Esteban Ocon': ocn31,
  'Lance Stroll': str18,
  'Gabriel Bortoleto': bor5,
  'Oliver Bearman': bea87,
  'Isack Hadjar': had6,
  'Arvid Lindblad': lin45,
  'Franco Colapinto': col43,
  'Carlos Sainz': sai55,
  }

 

  return (
    <div style={{ marginBottom: '2.5rem' }}>
      <p style={{
        fontFamily: 'var(--font-mono)',
        fontSize: '11px',
        color: 'var(--text-muted)',
        textTransform: 'uppercase',
        letterSpacing: '0.1em',
        marginBottom: '1.5rem',
      }}>Podium</p>

      <div style={{
        display: 'flex',
        alignItems: 'flex-end',
        justifyContent: 'center',
        gap: '4px',
        height: '320px',
      }}>
        {order.map((idx, i) => {
          const driver = podium[idx]
          if (!driver) return null
          const teamColor = teamColors[driver.team] || '#6b6b6b'
          const isFirst = idx === 0

          return (
            <div key={idx} style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              width: '200px',
            }}>
                <div style={{
  width: '120px',
  height: '120px',
  borderRadius: '50%',
  border: `3px solid ${teamColor}`,
  background: teamColor + '33',
  overflow: 'hidden',
  marginBottom: '12px',
  flexShrink: 0,
  display: 'flex',
  alignItems: 'flex-end',
  justifyContent: 'center',
}}>
                {driverPhotos[driver.driver] ? (
                  <img
                    src={driverPhotos[driver.driver]}
                    alt={driver.driver}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      objectPosition: 'top',
                    }}
                  />
                ) : (
                  <span style={{
                    fontFamily: 'var(--font-display)',
                    fontSize: '28px',
                    fontWeight: 700,
                    color: teamColor,
                    paddingBottom: '8px',
                  }}>
                    {driver.driver.split(' ').map(n => n[0]).join('')}
                  </span>
                )}
              </div>
                            

              <p style={{
                fontFamily: 'var(--font-display)',
                fontSize: isFirst ? '20px' : '17px',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                textAlign: 'center',
                marginBottom: '4px',
                color: isFirst ? '#fff' : 'var(--text-dim)',
              }}>
                {driver.driver}
              </p>

              {driver.actual_p && (
                <p style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '11px',
                  color: driver.exact_match_hit ? 'var(--green)' : 'var(--text-muted)',
                  marginBottom: '8px',
                }}>
                  {driver.exact_match_hit ? '✓ Actual: P' + driver.actual_p : 'Actual: P' + driver.actual_p}
                </p>
              )}

              <div style={{
                width: '100%',
                height: `${heights[i]}px`,
                background: 'var(--surface)',
                border: '1px solid var(--border)',
                borderBottom: 'none',
                borderRadius: '4px 4px 0 0',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column',
                gap: '4px',
                position: 'relative',
                overflow: 'hidden',
              }}>
                <div style={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  right: 0,
                  height: '4px',
                  background: teamColor,
                }} />
                <span style={{
                  fontFamily: 'var(--font-display)',
                  fontSize: '36px',
                  fontWeight: 700,
                  color: isFirst ? 'var(--red)' : 'var(--text-muted)',
                  lineHeight: 1,
                }}>
                  {labels[i]}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}