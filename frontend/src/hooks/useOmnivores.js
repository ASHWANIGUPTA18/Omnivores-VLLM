/**
 * The application store.
 *
 * All server state arrives as WebSocket events, so the store is a reducer over
 * the event protocol defined in `backend/app/core/events.py`. One switch,
 * one case per event type - which is the payoff for having defined that
 * protocol explicitly instead of sending ad-hoc JSON.
 *
 * The one subtlety is delta handling: `message.delta` mutates a message that
 * `message.created` introduced. Deltas can outrun their `created` frame under
 * load, so an orphan delta creates a placeholder rather than being dropped -
 * losing the opening words of an answer is a very visible bug.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react'
import { ConnectionState, OmnivoresSocket } from '../services/ws'
import { api } from '../services/api'

const MAX_TOASTS = 4

const initialState = {
  connection: ConnectionState.CONNECTING,
  config: null,

  channels: [],
  agents: [],
  activeChannelId: 'general',

  messagesByChannel: {},
  // channelId -> { authorId: displayName }
  typingByChannel: {},

  documents: [],
  // Client-side upload progress, keyed by a temporary id.
  uploads: {},

  toasts: [],
  loaded: false,
}

function upsertMessage(list, message) {
  const index = list.findIndex((m) => m.id === message.id)
  if (index === -1) return [...list, message]
  const next = list.slice()
  next[index] = { ...next[index], ...message }
  return next
}

function patchMessage(list, messageId, patch) {
  const index = list.findIndex((m) => m.id === messageId)
  if (index === -1) return null
  const next = list.slice()
  next[index] = { ...next[index], ...patch }
  return next
}

function withChannel(state, channelId, updater) {
  const current = state.messagesByChannel[channelId] || []
  const updated = updater(current)
  if (updated === null) return state
  return {
    ...state,
    messagesByChannel: { ...state.messagesByChannel, [channelId]: updated },
  }
}

function addToast(state, toast) {
  const entry = { id: `toast_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`, ...toast }
  return { ...state, toasts: [...state.toasts, entry].slice(-MAX_TOASTS) }
}

function reducer(state, action) {
  switch (action.type) {
    // --- local actions ---------------------------------------------------
    case 'connection':
      return { ...state, connection: action.state }

    case 'bootstrap':
      return {
        ...state,
        config: action.config ?? state.config,
        channels: action.channels ?? state.channels,
        agents: action.agents ?? state.agents,
        documents: action.documents ?? state.documents,
        loaded: true,
      }

    case 'selectChannel':
      return { ...state, activeChannelId: action.channelId }

    case 'setHistory':
      return {
        ...state,
        messagesByChannel: {
          ...state.messagesByChannel,
          [action.channelId]: action.messages,
        },
      }

    case 'uploadProgress':
      return {
        ...state,
        uploads: {
          ...state.uploads,
          [action.id]: { name: action.name, progress: action.progress },
        },
      }

    case 'uploadDone': {
      const uploads = { ...state.uploads }
      delete uploads[action.id]
      return { ...state, uploads }
    }

    case 'toast':
      return addToast(state, action.toast)

    case 'dismissToast':
      return { ...state, toasts: state.toasts.filter((t) => t.id !== action.id) }

    // --- server events ---------------------------------------------------
    case 'connection.ready':
      return {
        ...state,
        channels: action.data.channels || state.channels,
        agents: action.data.agents || state.agents,
      }

    case 'message.created': {
      const message = action.data.message
      return withChannel(state, message.channel_id, (list) => upsertMessage(list, message))
    }

    case 'message.delta': {
      const { message_id: id, channel_id: channelId, content } = action.data
      return withChannel(state, channelId, (list) => {
        const index = list.findIndex((m) => m.id === id)
        if (index === -1) {
          // The delta beat its `message.created`. Hold the text in a
          // placeholder; the created frame will merge into it.
          return [
            ...list,
            {
              id,
              channel_id: channelId,
              content,
              status: 'streaming',
              author: { id: 'agent', type: 'agent', display_name: 'Agent' },
              citations: [],
              tool_invocations: [],
            },
          ]
        }
        const next = list.slice()
        next[index] = { ...next[index], content: next[index].content + content, status: 'streaming' }
        return next
      })
    }

    case 'message.completed': {
      const message = action.data.message
      return withChannel(state, message.channel_id, (list) => upsertMessage(list, message))
    }

    case 'message.failed': {
      const { message_id: id, channel_id: channelId, code, error, retryable } = action.data
      const next = withChannel(state, channelId, (list) =>
        patchMessage(list, id, { status: 'failed', error: { code, message: error } })
      )
      return addToast(next, { level: 'error', code, message: error, retryable })
    }

    case 'agent.tool_started':
    case 'agent.tool_completed': {
      const { message_id: id, channel_id: channelId, invocation } = action.data
      return withChannel(state, channelId, (list) => {
        const index = list.findIndex((m) => m.id === id)
        if (index === -1) return null
        const message = list[index]
        const invocations = message.tool_invocations || []
        const position = invocations.findIndex((i) => i.id === invocation.id)
        const updated =
          position === -1
            ? [...invocations, invocation]
            : invocations.map((i, n) => (n === position ? invocation : i))
        const next = list.slice()
        next[index] = { ...message, tool_invocations: updated }
        return next
      })
    }

    case 'agent.completed': {
      const { message_id: id, channel_id: channelId, citations, metrics } = action.data
      return withChannel(state, channelId, (list) =>
        patchMessage(list, id, { citations: citations || [], metrics })
      )
    }

    case 'agent.failed':
      // `message.failed` follows and carries the same information to the UI.
      return state

    case 'typing.started': {
      const { channel_id: channelId, author_id: authorId, display_name: name } = action.data
      return {
        ...state,
        typingByChannel: {
          ...state.typingByChannel,
          [channelId]: { ...(state.typingByChannel[channelId] || {}), [authorId]: name },
        },
      }
    }

    case 'typing.stopped': {
      const { channel_id: channelId, author_id: authorId } = action.data
      const current = { ...(state.typingByChannel[channelId] || {}) }
      delete current[authorId]
      return {
        ...state,
        typingByChannel: { ...state.typingByChannel, [channelId]: current },
      }
    }

    case 'channel.history':
      return {
        ...state,
        messagesByChannel: {
          ...state.messagesByChannel,
          [action.data.channel_id]: action.data.messages || [],
        },
      }

    case 'document.uploaded':
    case 'document.processing':
    case 'document.ready':
    case 'document.failed': {
      const document = action.data.document
      const index = state.documents.findIndex((d) => d.id === document.id)
      const documents =
        index === -1
          ? [document, ...state.documents]
          : state.documents.map((d, n) => (n === index ? document : d))

      const next = { ...state, documents }
      if (action.type === 'document.failed') {
        return addToast(next, {
          level: 'error',
          message: `${document.filename}: ${document.error || 'processing failed'}`,
        })
      }
      return next
    }

    case 'document.deleted':
      return {
        ...state,
        documents: state.documents.filter((d) => d.id !== action.data.document_id),
      }

    case 'error':
      return addToast(state, {
        level: 'error',
        code: action.data.code,
        message: action.data.message,
        retryable: action.data.retryable,
      })

    case 'pong':
      return state

    default:
      return state
  }
}

export function useOmnivores() {
  const [state, dispatch] = useReducer(reducer, initialState)
  const socketRef = useRef(null)

  // --- socket lifecycle ----------------------------------------------------
  useEffect(() => {
    const socket = new OmnivoresSocket({
      onEvent: (event) => dispatch({ type: event.type, data: event.data }),
      onStateChange: (connection) => dispatch({ type: 'connection', state: connection }),
    })
    socketRef.current = socket
    socket.connect()
    return () => socket.disconnect()
  }, [])

  // --- initial load --------------------------------------------------------
  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        const [config, channels, agents, documents] = await Promise.all([
          api.config(),
          api.channels(),
          api.agents(),
          api.documents(),
        ])
        if (cancelled) return
        dispatch({
          type: 'bootstrap',
          config,
          channels: channels.channels,
          agents: agents.agents,
          documents: documents.documents,
        })
      } catch (error) {
        if (cancelled) return
        dispatch({
          type: 'toast',
          toast: { level: 'error', message: error.message, retryable: true },
        })
        dispatch({ type: 'bootstrap' })
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [])

  // --- history on channel change / reconnect -------------------------------
  useEffect(() => {
    if (state.connection !== ConnectionState.CONNECTED) return
    socketRef.current?.subscribe(state.activeChannelId)
  }, [state.activeChannelId, state.connection])

  // --- actions -------------------------------------------------------------
  const sendMessage = useCallback((content, documentIds = []) => {
    const text = content.trim()
    if (!text) return false
    return socketRef.current?.sendMessage(state.activeChannelId, text, documentIds) ?? false
  }, [state.activeChannelId])

  const selectChannel = useCallback((channelId) => {
    dispatch({ type: 'selectChannel', channelId })
  }, [])

  const setTyping = useCallback((active) => {
    socketRef.current?.setTyping(state.activeChannelId, active)
  }, [state.activeChannelId])

  const cancelMessage = useCallback((messageId) => {
    socketRef.current?.cancel(messageId)
  }, [])

  const uploadDocument = useCallback(async (file) => {
    const uploadId = `upload_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`
    dispatch({ type: 'uploadProgress', id: uploadId, name: file.name, progress: 0 })
    try {
      await api.uploadDocument(file, {
        channelId: state.activeChannelId,
        onProgress: (progress) =>
          dispatch({ type: 'uploadProgress', id: uploadId, name: file.name, progress }),
      })
    } catch (error) {
      dispatch({ type: 'toast', toast: { level: 'error', message: error.message } })
    } finally {
      dispatch({ type: 'uploadDone', id: uploadId })
    }
  }, [state.activeChannelId])

  const deleteDocument = useCallback(async (documentId) => {
    try {
      await api.deleteDocument(documentId)
    } catch (error) {
      dispatch({ type: 'toast', toast: { level: 'error', message: error.message } })
    }
  }, [])

  const dismissToast = useCallback((id) => dispatch({ type: 'dismissToast', id }), [])

  // --- derived -------------------------------------------------------------
  const messages = useMemo(
    () => state.messagesByChannel[state.activeChannelId] || [],
    [state.messagesByChannel, state.activeChannelId]
  )

  const activeChannel = useMemo(
    () => state.channels.find((c) => c.id === state.activeChannelId) || null,
    [state.channels, state.activeChannelId]
  )

  const channelAgents = useMemo(() => {
    const allowed = activeChannel?.agents
    if (!allowed || allowed.length === 0) return state.agents
    return state.agents.filter((a) => allowed.includes(a.id))
  }, [state.agents, activeChannel])

  const channelDocuments = useMemo(
    () =>
      state.documents.filter(
        (d) => !d.channel_id || d.channel_id === state.activeChannelId
      ),
    [state.documents, state.activeChannelId]
  )

  const typingNames = useMemo(
    () => Object.values(state.typingByChannel[state.activeChannelId] || {}),
    [state.typingByChannel, state.activeChannelId]
  )

  return {
    ...state,
    messages,
    activeChannel,
    channelAgents,
    channelDocuments,
    typingNames,
    isConnected: state.connection === ConnectionState.CONNECTED,
    sendMessage,
    selectChannel,
    setTyping,
    cancelMessage,
    uploadDocument,
    deleteDocument,
    dismissToast,
  }
}
