import { useState, useEffect } from 'react'
import { getModelInfo } from '../api'

function DashboardPage() {
  const [info, setInfo] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchInfo()
  }, [])

  const fetchInfo = async () => {
    try {
      const res = await getModelInfo()
      setInfo(res)
    } catch (err) {
      setError(`Failed to load model info: ${err.message}. Make sure the Flask API is running.`)
    } finally {
      setLoading(false)
    }
  }

  const getBestMetric = (metricKey, lower = true) => {
    if (!info?.all_metrics) return null
    const values = info.all_metrics.map(m => m[metricKey])
    return lower ? Math.min(...values) : Math.max(...values)
  }

  return (
    <div className="fade-in">
      <div className="page-header">
        <h2>Model Dashboard</h2>
        <p>Compare XGBoost, LightGBM, and Random Forest performance metrics</p>
      </div>

      {loading && (
        <div className="card" style={{ textAlign: 'center', padding: 48 }}>
          <div className="loading-spinner" style={{ width: 32, height: 32, margin: '0 auto 16px' }}></div>
          <div style={{ color: 'var(--text-secondary)' }}>Loading model data...</div>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      {info && (
        <>
          {/* Top metrics */}
          <div className="grid-3" style={{ marginBottom: 32 }}>
            <div className="metric-card">
              <div className="metric-label">Best Model</div>
              <div className="metric-value" style={{ fontSize: 32 }}>
                {info.best_model || 'N/A'}
              </div>
              <div className="metric-sublabel">Selected by lowest MAE</div>
              {info.best_model && (
                <div className="active-badge">
                  <div className="active-badge-dot"></div>
                  ACTIVE
                </div>
              )}
            </div>
            <div className="metric-card">
              <div className="metric-label">Best MAE</div>
              <div className="metric-value">
                {info.best_metrics?.['MAE (hours)'] ?? 'N/A'}
                <span style={{ fontSize: 16, color: 'var(--text-secondary)' }}>h</span>
              </div>
              <div className="metric-sublabel">Mean Absolute Error</div>
            </div>
            <div className="metric-card">
              <div className="metric-label">Best R²</div>
              <div className="metric-value">
                {info.best_metrics?.['R²'] ?? 'N/A'}
              </div>
              <div className="metric-sublabel">Variance Explained</div>
            </div>
          </div>

          {/* Comparison Table */}
          <div className="card" style={{ marginBottom: 32 }}>
            <div className="card-title">
              <span className="card-icon">📊</span>
              Model Comparison (Regression)
            </div>

            {info.all_metrics ? (
              <div style={{ overflowX: 'auto' }}>
                <table className="comparison-table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>MAE (hours)</th>
                      <th>RMSE (hours)</th>
                      <th>R²</th>
                      <th>MAPE (%)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {info.all_metrics.map((m, i) => {
                      const isBest = m.Model === info.best_model
                      return (
                        <tr key={i} className={isBest ? 'best-row' : ''}>
                          <td>
                            {m.Model}
                            {isBest && <span className="best-badge">BEST</span>}
                          </td>
                          <td className={m['MAE (hours)'] === getBestMetric('MAE (hours)', true) ? 'best-metric' : ''}>
                            {m['MAE (hours)']}
                          </td>
                          <td className={m['RMSE (hours)'] === getBestMetric('RMSE (hours)', true) ? 'best-metric' : ''}>
                            {m['RMSE (hours)']}
                          </td>
                          <td className={m['R²'] === getBestMetric('R²', false) ? 'best-metric' : ''}>
                            {m['R²']}
                          </td>
                          <td className={m['MAPE (%)'] === getBestMetric('MAPE (%)', true) ? 'best-metric' : ''}>
                            {m['MAPE (%)']}%
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: 32, color: 'var(--text-muted)' }}>
                No model comparison data. Run main.py first.
              </div>
            )}
          </div>

          {/* Visual comparison bars */}
          {info.all_metrics && (
            <div className="grid-2" style={{ marginBottom: 32 }}>
              <div className="card">
                <div className="card-title">
                  <span className="card-icon">🎯</span>
                  MAE Comparison (lower is better)
                </div>
                {info.all_metrics.map((m, i) => {
                  const maxMAE = Math.max(...info.all_metrics.map(x => x['MAE (hours)']))
                  const pct = (m['MAE (hours)'] / maxMAE) * 100
                  const barClass = m.Model.toLowerCase().includes('xgb') ? 'dash-bar-xgboost' :
                                   m.Model.toLowerCase().includes('lightgbm') ? 'dash-bar-lightgbm' : 'dash-bar-randomforest'
                  return (
                    <div className="dash-bar-container" key={i}>
                      <div className="dash-bar-header">
                        <span className="dash-bar-name">{m.Model}</span>
                        <span className="dash-bar-value">{m['MAE (hours)']}h</span>
                      </div>
                      <div className="dash-bar-track">
                        <div 
                          className={`dash-bar-fill ${barClass}`}
                          style={{ width: `${pct}%`, animationDelay: `${i * 100}ms` }}
                        ></div>
                      </div>
                    </div>
                  )
                })}
              </div>

              <div className="card">
                <div className="card-title">
                  <span className="card-icon">📈</span>
                  R² Comparison (higher is better)
                </div>
                {info.all_metrics.map((m, i) => {
                  const pct = m['R²'] * 100
                  const barClass = m.Model.toLowerCase().includes('xgb') ? 'dash-bar-xgboost' :
                                   m.Model.toLowerCase().includes('lightgbm') ? 'dash-bar-lightgbm' : 'dash-bar-randomforest'
                  return (
                    <div className="dash-bar-container" key={i}>
                      <div className="dash-bar-header">
                        <span className="dash-bar-name">{m.Model}</span>
                        <span className="dash-bar-value">{m['R²']}</span>
                      </div>
                      <div className="dash-bar-track">
                        <div 
                          className={`dash-bar-fill ${barClass}`}
                          style={{ width: `${pct}%`, animationDelay: `${i * 100}ms` }}
                        ></div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Global Explainability */}
          <div className="card" style={{ marginBottom: 32 }}>
            <div className="card-title">
              <span className="card-icon">🔍</span>
              Global Explainability (SHAP Summary)
            </div>
            <div className="shap-image-container">
              <img 
                src="http://localhost:5000/api/plots/shap_summary.png" 
                alt="SHAP Summary Plot" 
                onError={(e) => { e.target.style.display = 'none'; e.target.nextSibling.style.display = 'block'; }}
              />
              <div style={{ display: 'none', color: 'var(--text-muted)', padding: 32 }}>
                SHAP plot not found. Ensure you have run the full training pipeline to generate explainability plots.
              </div>
            </div>
            <div className="shap-read-guide">
              <strong>How to read this chart:</strong> This chart ranks features by their overall impact on the model's predictions. 
              Features at the top are the most important. The colors show whether a high value (red) or low value (blue) 
              for that feature increases or decreases the predicted scent duration.
            </div>
          </div>

          {/* Metric descriptions */}
          <div className="card">
            <div className="card-title">
              <span className="card-icon">📖</span>
              Understanding the Metrics
            </div>
            <div className="metrics-understanding-grid">
              <div className="metric-explain-card">
                <div className="metric-explain-icon">📏</div>
                <div className="metric-explain-name">MAE (Mean Absolute Error)</div>
                <div className="metric-explain-desc">
                  Average prediction error in hours. A MAE of 1.5 means predictions are off by 1.5 hours on average.
                </div>
              </div>
              <div className="metric-explain-card">
                <div className="metric-explain-icon">⚡</div>
                <div className="metric-explain-name">RMSE (Root Mean Squared Error)</div>
                <div className="metric-explain-desc">
                  Like MAE but penalizes large errors more. Lower is better.
                </div>
              </div>
              <div className="metric-explain-card">
                <div className="metric-explain-icon">🎯</div>
                <div className="metric-explain-name">R² (Coefficient of Determination)</div>
                <div className="metric-explain-desc">
                  Proportion of variance explained. 1.0 = perfect, 0.0 = baseline. Values above 0.85 indicate strong predictions.
                </div>
              </div>
              <div className="metric-explain-card">
                <div className="metric-explain-icon">⚖️</div>
                <div className="metric-explain-name">MAPE (Mean Absolute % Error)</div>
                <div className="metric-explain-desc">
                  Prediction error as a percentage. Useful for comparing across different duration ranges.
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default DashboardPage
