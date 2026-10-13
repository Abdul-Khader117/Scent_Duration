import { useState } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import PredictionPage from './pages/PredictionPage'
import OptimizerPage from './pages/OptimizerPage'
import DashboardPage from './pages/DashboardPage'
import './index.css'

function App() {
  return (
    <BrowserRouter>
      {/* Floating background particles */}
      <div className="particles-container">
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
        <div className="particle"></div>
      </div>

      <div className="app">
        <aside className="sidebar">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon">&#128172;</div>
            <h1>
              ScentAI
              <span>Fragrance Intelligence Engine</span>
            </h1>
          </div>

          <nav className="sidebar-nav">
            <NavLink
              to="/"
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
              end
            >
              <span className="nav-icon">⚗</span>
              <span>Predict Duration</span>
            </NavLink>

            <NavLink
              to="/optimize"
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
            >
              <span className="nav-icon">✦</span>
              <span>Optimize Ingredients</span>
            </NavLink>

            <NavLink
              to="/dashboard"
              className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}
            >
              <span className="nav-icon">◧</span>
              <span>Model Dashboard</span>
            </NavLink>
          </nav>
        </aside>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<PredictionPage />} />
            <Route path="/optimize" element={<OptimizerPage />} />
            <Route path="/dashboard" element={<DashboardPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
