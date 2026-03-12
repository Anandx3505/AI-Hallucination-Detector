import React from 'react';
import Hero from './components/Hero';
import Pipeline from './components/Pipeline';
import Comparison from './components/Comparison';
import Demo from './components/Demo';
import Team from './components/Team';
import './index.css';

function App() {
  return (
    <div className="app-container">
      {/* Navbar Minimalist */}
      <nav style={{
        position: 'fixed', top: 0, left: 0, right: 0,
        height: '64px', display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'rgba(255, 255, 255, 0.8)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        borderBottom: '1px solid rgba(0,0,0,0.05)',
        zIndex: 50
      }}>
        <div style={{
          width: '100%', maxWidth: '1200px', padding: '0 2rem',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center'
        }}>
          <div style={{ fontWeight: 600, fontSize: '1.1rem', letterSpacing: '-0.02em', color: '#1d1d1f' }}>
            Halluci<span style={{ color: '#0066cc' }}>Detect</span>
          </div>
          <div style={{ display: 'flex', gap: '2rem', fontSize: '0.9rem', color: '#86868b' }}>
            <a href="#pipeline" style={{ color: 'inherit', textDecoration: 'none' }}>Pipeline</a>
            <a href="#comparison" style={{ color: 'inherit', textDecoration: 'none' }}>Results</a>
            <a href="#demo" style={{ color: 'inherit', textDecoration: 'none' }}>Demo</a>
            <a href="#team" style={{ color: 'inherit', textDecoration: 'none' }}>Team</a>
          </div>
        </div>
      </nav>

      <main style={{ paddingTop: '64px' }}>
        <Hero />
        <Pipeline />
        <Comparison />
        <Demo />
        <Team />
      </main>

      {/* Minimal Footer */}
      <footer style={{
        padding: '3rem 0',
        textAlign: 'center',
        background: '#f5f5f7',
        color: '#86868b',
        fontSize: '0.85rem'
      }}>
        <p>Explainable Graph-Based AI Hallucination Detection &mdash; Major Project 2026</p>
      </footer>
    </div>
  );
}

export default App;
