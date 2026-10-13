const API_BASE = 'http://localhost:5000/api'

export async function predict(data) {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export async function getIngredients() {
  const res = await fetch(`${API_BASE}/ingredients`)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export async function getOptions() {
  const res = await fetch(`${API_BASE}/options`)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export async function getModelInfo() {
  const res = await fetch(`${API_BASE}/model-info`)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export async function getOptimalCombinations(limit = 15) {
  const res = await fetch(`${API_BASE}/optimize?limit=${limit}`)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}
