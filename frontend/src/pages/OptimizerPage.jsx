import { useState, useEffect } from 'react'
import { getOptimalCombinations } from '../api'

function OptimizerPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      const res = await getOptimalCombinations(25)
      setData(res)
    } catch (err) {
      setError(`Failed to load optimization data: ${err.message}. Make sure the Flask API is running.`)
    } finally {
      setLoading(false)
    }
  }

  const getCategoryColor = (cat) => {
    const map = {
      'Short': 'var(--accent-rose)',
      'Medium': 'var(--accent-amber)',
      'Long': 'var(--accent-teal)',
      'Very Long': 'var(--accent-primary)',
    }
    return map[cat] || 'var(--text-primary)'
  }

  return (
    <div className="fade-in">
      <div className="page-header">
        <h2>Ingredient Optimizer</h2>
        <p>Discover the best ingredient combinations for maximum scent longevity</p>
      </div>

      {loading && (
        <div className="card" style={{ textAlign: 'center', padding: 48 }}>
          <div className="loading-spinner" style={{ width: 32, height: 32, margin: '0 auto 16px' }}></div>
          <div style={{ color: 'var(--text-secondary)' }}>Loading optimization results...</div>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      {data && (
        <div className="grid-layout">
          {/* Left: Top Combinations */}
          <div>
            <div className="card">
              <div className="card-title">
                <span className="card-icon">🏆</span>
                Top Predicted Ingredient Combinations
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
                Model predictions at 15% concentration (Eau de Parfum)
              </div>

              {data.optimal_combinations?.map((combo, i) => {
                const isTop3 = i < 3
                let rankClass = 'rank-other'
                let rankText = `#${i + 1}`
                if (i === 0) { rankClass = 'rank-1'; rankText = '1st' }
                else if (i === 1) { rankClass = 'rank-2'; rankText = '2nd' }
                else if (i === 2) { rankClass = 'rank-3'; rankText = '3rd' }

                return (
                  <div className={`combo-card ${isTop3 ? 'top-3-card' : ''}`} key={i}>
                    <div className={`combo-rank ${rankClass}`}>
                      {rankText}
                    </div>
                    <div className="combo-info">
                      <div className="combo-ingredients">{combo.Ingredients}</div>
                      <div className="combo-meta">
                        {combo.NumIngredients} ingredients |
                        Top: {combo.TopNote_Count} | Mid: {combo.MiddleNote_Count} | Base: {combo.BaseNote_Count}
                      </div>
                    </div>
                    <div className="combo-duration" style={{ color: getCategoryColor(combo.Predicted_Category) }}>
                      {combo.Predicted_Duration_Hours}h
                    </div>
                  </div>
                )
              })}

              {(!data.optimal_combinations || data.optimal_combinations.length === 0) && (
                <div style={{ textAlign: 'center', padding: 32, color: 'var(--text-muted)' }}>
                  No optimization data available. Run main.py first.
                </div>
              )}
            </div>
          </div>

          {/* Right: Ingredient Contributions */}
          <div>
            <div className="card">
              <div className="card-title">
                <span className="card-icon">📊</span>
                Ingredient Impact (SHAP)
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 16 }}>
                How much each ingredient contributes to scent duration prediction
              </div>

              {data.ingredient_contributions && Object.keys(data.ingredient_contributions).length > 0 ? (
                Object.entries(data.ingredient_contributions).map(([ing, score], i) => {
                  const maxScore = Math.max(...Object.values(data.ingredient_contributions))
                  const pct = (score / maxScore) * 100
                  const noteType = ['Bergamot','Lemon','Mandarin','Grapefruit','Orange','Lavender','Mint','Basil','Neroli','Petitgrain'].includes(ing) ? 'top' :
                    ['Rose','Jasmine','Ylang-ylang','Geranium','Iris','Lily','Tuberose','Violet','Freesia','Gardenia'].includes(ing) ? 'middle' : 'base'

                  return (
                    <div className="shap-bar-container" key={ing}>
                      <div className="shap-bar-header">
                        <span className="shap-bar-name">
                          {ing}
                          <span className={`shap-bar-note-badge ${noteType}`}>
                            {noteType}
                          </span>
                        </span>
                        <span className="shap-bar-score">{score.toFixed(4)}</span>
                      </div>
                      <div className="shap-bar-track">
                        <div 
                          className={`shap-bar-fill animate ${noteType}`}
                          style={{ width: `${pct}%`, animationDelay: `${i * 50}ms` }}
                        ></div>
                      </div>
                    </div>
                  )
                })
              ) : (
                <div style={{ textAlign: 'center', padding: 32, color: 'var(--text-muted)' }}>
                  No ingredient contribution data available.
                </div>
              )}
            </div>

            <div className="card" style={{ marginTop: 20 }}>
              <div className="card-title">
                <span className="card-icon">🔬</span>
                Key Findings
              </div>
              <ul className="findings-list">
                <li><span className="finding-icon">🌟</span> Base note ingredients (sandalwood, musk) are strongest fixatives</li>
                <li><span className="finding-icon">📈</span> Combinations with 2-3 base notes achieve longest duration</li>
                <li><span className="finding-icon">⚖️</span> 4-6 ingredients is the optimal complexity range</li>
                <li><span className="finding-icon">⏱️</span> The model predicts duration in original hours scale</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default OptimizerPage
