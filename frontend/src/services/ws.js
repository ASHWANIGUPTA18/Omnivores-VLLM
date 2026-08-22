/**
 * WebSocket client with reconnection and heartbeat.
 *
 * Three things this handles that a bare `new WebSocket(...)` does not:
 *
 * 1. **Reconnection with backoff and jitter.** A backend restart should not
 *    require a page reload, and without jitter every open tab would reconnect
 *    in lockstep and hammer the server the moment it comes back.
 * 2. **Heartbeats.** Proxies silently drop idle WebSockets, which otherwise
 *    presents as an app that mysteriously stops updating.
 * 3. **A send queue.** Frames sent while the socket is down are held and
 *    flushed on reconnect instead of being thrown away.
 */

const WS_URL = import.meta.env.VITE_WS_URL || ''
const API_KEY = import.meta.env.VITE_API_KEY || ''

const HEARTBEAT_MS = 25000
const BASE_BACKOFF_MS = 500
const MAX_BACKOFF_MS = 15000
//: Frames buffered while disconnected. Beyond this the user is far better
//: served by an error than by a flood of stale messages on reconnect.
const MAX_QUEUE = 50

function defaultUrl() {
  if (WS_URL) return WS_URL
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/ws`
}

export const ConnectionState = {
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  RECONNECTING: 'reconnecting',
  DISCONNECTED: 'disconnected',
}

export class OmnivoresSocket {
  constructor({ onEvent, onStateChange } = {}) {
    this.onEvent = onEvent || (() => {})
    this.onStateChange = onStateChange || (() => {})

    this.socket = null
    this.state = ConnectionState.DISCONNECTED
    this.attempt = 0
    this.queue = []
    this.closedByUs = false

    this.reconnectTimer = null
    this.heartbeatTimer = null
  }

  connect() {
    if (this.socket && this.socket.readyState <= WebSocket.OPEN) return

    this.closedByUs = false
    this.#setState(this.attempt === 0 ? ConnectionState.CONNECTING : ConnectionState.RECONNECTING)

    const url = new URL(defaultUrl())
    if (API_KEY) url.searchParams.set('token', API_KEY)

    try {
      this.socket = new WebSocket(url.toString())
    } catch {
      this.#scheduleReconnect()
      return
    }

    this.socket.onopen = () => {
      this.attempt = 0
      this.#setState(ConnectionState.CONNECTED)
      this.#startHeartbeat()
      this.#flushQueue()
    }

    this.socket.onmessage = (raw) => {
      let event
      try {
        event = JSON.parse(raw.data)
      } catch {
        return
      }
      if (event?.type) this.onEvent(event)
    }

    this.socket.onclose = () => {
      this.#stopHeartbeat()
      this.socket = null
      if (this.closedByUs) {
        this.#setState(ConnectionState.DISCONNECTED)
      } else {
        this.#scheduleReconnect()
      }
    }

    // An error is always followed by a close, which is where reconnection is
    // handled; doing it here too would double-schedule.
    this.socket.onerror = () => {}
  }

  disconnect() {
    this.closedByUs = true
    this.#stopHeartbeat()
    clearTimeout(this.reconnectTimer)
    this.queue = []
    if (this.socket) {
      this.socket.close(1000, 'client closing')
      this.socket = null
    }
    this.#setState(ConnectionState.DISCONNECTED)
  }

  send(type, data = {}) {
    const frame = JSON.stringify({ type, data })
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(frame)
      return true
    }
    if (this.queue.length < MAX_QUEUE) this.queue.push(frame)
    return false
  }

  sendMessage(channelId, content, documentIds = []) {
    return this.send('client.send_message', {
      channel_id: channelId,
      content,
      document_ids: documentIds,
    })
  }

  subscribe(channelId, limit = 100) {
    return this.send('client.subscribe', { channel_id: channelId, limit })
  }

  setTyping(channelId, active) {
    return this.send('client.typing', { channel_id: channelId, active })
  }

  cancel(messageId) {
    return this.send('client.cancel', { message_id: messageId })
  }

  get isConnected() {
    return this.state === ConnectionState.CONNECTED
  }

  // --- internals ---------------------------------------------------------

  #setState(state) {
    if (this.state === state) return
    this.state = state
    this.onStateChange(state)
  }

  #flushQueue() {
    const pending = this.queue
    this.queue = []
    for (const frame of pending) {
      if (this.socket?.readyState === WebSocket.OPEN) this.socket.send(frame)
    }
  }

  #scheduleReconnect() {
    this.#setState(ConnectionState.RECONNECTING)
    // Exponential backoff with jitter, so many tabs do not retry in lockstep.
    const backoff = Math.min(BASE_BACKOFF_MS * 2 ** this.attempt, MAX_BACKOFF_MS)
    const delay = backoff * (0.5 + Math.random() * 0.5)
    this.attempt += 1

    clearTimeout(this.reconnectTimer)
    this.reconnectTimer = setTimeout(() => this.connect(), delay)
  }

  #startHeartbeat() {
    this.#stopHeartbeat()
    this.heartbeatTimer = setInterval(() => {
      this.send('client.ping', { ts: Date.now() })
    }, HEARTBEAT_MS)
  }

  #stopHeartbeat() {
    clearInterval(this.heartbeatTimer)
    this.heartbeatTimer = null
  }
}
