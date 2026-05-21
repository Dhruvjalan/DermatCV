const API = 'http://localhost:8000/api'

export async function upload(userName, file) {
  const form = new FormData()
  form.append('user_name', userName)
  form.append('file', file)
  const res = await fetch(`${API}/upload`, { method: 'POST', body: form })
  return res.json()
}

export async function analyze(imageId) {
  const res = await fetch(`${API}/analyze/${imageId}`, { method: 'POST' })
  return res.json()
}

export async function save(resultId) {
  const res = await fetch(`${API}/results/${resultId}/save`, { method: 'POST' })
  return res.json()
}

export async function history(userId) {
  const res = await fetch(`${API}/history/${userId}`)
  return res.json()
}

export async function admin() {
  const res = await fetch(`${API}/admin/records`)
  return res.json()
}
