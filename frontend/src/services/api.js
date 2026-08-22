/**
 * REST client.
 *
 * Errors from the backend arrive as `{ error: { code, message, retryable } }`.
 * Rather than let callers each unpack that, every failure here becomes an
 * `ApiError` carrying the same fields, so the UI can show the server's own
 * message instead of a generic "request failed".
 */

const BASE = import.meta.env.VITE_API_URL || ''
const API_KEY = import.meta.env.VITE_API_KEY || ''

export class ApiError extends Error {
  constructor(message, { code = 'unknown', status = 0, retryable = false } = {}) {
    super(message)
    this.name = 'ApiError'
    this.code = code
    this.status = status
    this.retryable = retryable
  }
}

function authHeaders(extra = {}) {
  return API_KEY ? { ...extra, Authorization: `Bearer ${API_KEY}` } : extra
}

async function unwrap(response) {
  if (response.ok) {
    return response.status === 204 ? null : response.json()
  }

  let payload = null
  try {
    payload = await response.json()
  } catch {
    // A non-JSON error body (a proxy error page, say) is not worth parsing.
  }

  const error = payload?.error
  throw new ApiError(error?.message || `Request failed (${response.status})`, {
    code: error?.code,
    status: response.status,
    retryable: error?.retryable ?? response.status >= 500,
  })
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, {
      ...options,
      headers: authHeaders(options.headers),
    })
  } catch (cause) {
    // fetch only rejects on a transport failure, which always means the
    // backend is unreachable rather than unhappy.
    throw new ApiError('Cannot reach the Omnivores backend. Is it running?', {
      code: 'network_error',
      retryable: true,
    })
  }
  return unwrap(response)
}

export const api = {
  config: () => request('/config'),
  health: () => request('/health/full'),
  metrics: () => request('/health/metrics'),

  channels: () => request('/api/channels'),
  messages: (channelId, limit = 100) =>
    request(`/api/channels/${channelId}/messages?limit=${limit}`),

  agents: () => request('/api/agents'),

  documents: (channelId) =>
    request(`/api/documents${channelId ? `?channel_id=${encodeURIComponent(channelId)}` : ''}`),

  deleteDocument: (id) => request(`/api/documents/${id}`, { method: 'DELETE' }),
  reindexDocument: (id) => request(`/api/documents/${id}/reindex`, { method: 'POST' }),
  uploadLimits: () => request('/api/documents/limits/info'),

  /**
   * Uploads via XMLHttpRequest rather than fetch, because fetch still cannot
   * report upload progress and a 50 MB PDF needs a progress bar.
   */
  uploadDocument(file, { channelId, onProgress } = {}) {
    return new Promise((resolve, reject) => {
      const form = new FormData()
      form.append('file', file)
      if (channelId) form.append('channel_id', channelId)

      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE}/api/documents`)
      if (API_KEY) xhr.setRequestHeader('Authorization', `Bearer ${API_KEY}`)

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress(event.loaded / event.total)
        }
      }

      xhr.onload = () => {
        let payload = null
        try {
          payload = JSON.parse(xhr.responseText)
        } catch {
          payload = null
        }
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(payload)
        } else {
          reject(
            new ApiError(payload?.error?.message || `Upload failed (${xhr.status})`, {
              code: payload?.error?.code,
              status: xhr.status,
            })
          )
        }
      }

      xhr.onerror = () =>
        reject(new ApiError('Upload failed - the backend is unreachable.', {
          code: 'network_error',
          retryable: true,
        }))

      xhr.send(form)
    })
  },
}
