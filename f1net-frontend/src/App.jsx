import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Predict from './pages/Predict'
import Compare from './pages/Compare'
import Admin from './pages/Admin'



export default function App() {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)' }}>
      <Navbar />
      <Routes>
        <Route path="/" element={<Predict />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/admin" element={<Admin />} />
      </Routes>
    </div>
  )
}