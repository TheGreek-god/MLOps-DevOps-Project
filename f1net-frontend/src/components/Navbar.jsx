import { Link, useLocation } from 'react-router-dom'

export default function Navbar() {
  const { pathname } = useLocation()

  const linkStyle = (path) => ({
    fontFamily: 'var(--font-display)',
    fontSize: '15px',
    fontWeight: 600,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    color: pathname === path ? 'var(--red)' : 'var(--text-muted)',
    transition: 'color 0.15s',
    padding: '4px 0',
    borderBottom: pathname === path ? '2px solid var(--red)' : '2px solid transparent',
  })

  return (
    <nav style={{
      background: 'var(--surface)',
      borderBottom: '1px solid var(--border)',
      padding: '0 2rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      height: '56px',
      position: 'sticky',
      top: 0,
      zIndex: 100,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <span style={{
          fontFamily: 'var(--font-display)',
          fontSize: '22px',
          fontWeight: 700,
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
        }}>
          F1<span style={{ color: 'var(--red)' }}>NET</span>
        </span>
      </div>
      <div style={{ display: 'flex', gap: '2rem' }}>
        <Link to="/" style={linkStyle('/')}>Predict</Link>
        <Link to="/compare" style={linkStyle('/compare')}>Compare</Link>
        <Link to="/admin" style={linkStyle('/admin')}>Admin</Link>
      </div>
    </nav>
  )
}