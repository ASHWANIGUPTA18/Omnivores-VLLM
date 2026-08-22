/**
 * The application shell.
 *
 * Layout: channels and documents on the left, the conversation in the middle,
 * a status header on top. Autoscroll follows the stream only while the user is
 * already at the bottom - yanking the viewport away from someone who scrolled
 * up to read a citation is the single most irritating thing a chat UI can do.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Composer } from './components/Composer'
import { Message } from './components/Message'
import { Sidebar } from './components/Sidebar'
import { useOmnivores } from './hooks/useOmnivores'
import { api } from './services/api'

const CONNECTION_LABELS = {
  connecting: { text: 'Connecting', tone: 'warn' },
  connected: { text: 'Connected', tone: 'ok' },
  reconnecting: { text: 'Reconnecting', tone: 'warn' },
  disconnected: { text: 'Disconnected', tone: 'bad' },
}

function Header({ channel, connection, config, typingNames }) {
  const status = CONNECTION_LABELS[connection] || CONNECTION_LABELS.disconnected
  const isMock = config?.inference?.provider === 'mock'

  return (
    <header className="chat-header">
      <div className="chat-title">
        <span className="channel-hash">#</span>
        <h1>{channel?.name || 'channel'}</h1>
        {channel?.description && <p>{channel.description}</p>}
      </div>

      <div className="chat-status">
        {typingNames.length > 0 && (
          <span className="typing-indicator">
            {typingNames.join(', ')} {typingNames.length === 1 ? 'is' : 'are'} typing
            <span className="dots" />
          </span>
        )}

        {config && (
          <span className={`model-badge ${isMock ? 'model-mock' : ''}`} title={
            isMock
              ? 'Responses are synthetic. Set LLM_PROVIDER=vllm for a real model.'
              : `${config.inference.provider} · ${config.inference.model}`
          }>
            {isMock ? 'MOCK MODEL' : config.inference.model}
          </span>
        )}

        <span className={`connection-badge connection-${status.tone}`}>
          <span className="connection-dot" />
          {status.text}
        </span>
      </div>
    </header>
  )
}

function Toasts({ toasts, onDismiss }) {
  if (!toasts.length) return null
  return (
    <div className="toasts">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast toast-${toast.level}`}>
          <div className="toast-body">
            {toast.code && <strong>{toast.code}</strong>}
            <span>{toast.message}</span>
          </div>
          <button type="button" onClick={() => onDismiss(toast.id)}>
            ✕
          </button>
        </div>
      ))}
    </div>
  )
}

function EmptyChannel({ agents, channelName }) {
  return (
    <div className="empty-channel">
      <h2>#{channelName}</h2>
      <p>Mention an agent to start. Upload a document first if you want it consulted.</p>
      <div className="empty-agents">
        {agents.map((agent) => (
          <div key={agent.id} className="empty-agent">
            <span className="agent-avatar" style={{ background: agent.color }}>
              {agent.avatar}
            </span>
            <div>
              <code>@{agent.id}</code>
              <p>{agent.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const {
    connection,
    config,
    channels,
    activeChannelId,
    activeChannel,
    channelAgents,
    channelDocuments,
    messages,
    typingNames,
    uploads,
    toasts,
    isConnected,
    loaded,
    sendMessage,
    selectChannel,
    setTyping,
    cancelMessage,
    uploadDocument,
    deleteDocument,
    dismissToast,
  } = useOmnivores()

  const [limits, setLimits] = useState(null)
  const [scopedDocumentIds, setScopedDocumentIds] = useState([])
  const scrollRef = useRef(null)
  const pinnedToBottom = useRef(true)

  useEffect(() => {
    api.uploadLimits().then(setLimits).catch(() => setLimits(null))
  }, [])

  // Clearing the scope on channel change avoids silently carrying a document
  // filter into a conversation where it makes no sense.
  useEffect(() => setScopedDocumentIds([]), [activeChannelId])

  const handleScroll = useCallback(() => {
    const node = scrollRef.current
    if (!node) return
    const distanceFromBottom = node.scrollHeight - node.scrollTop - node.clientHeight
    pinnedToBottom.current = distanceFromBottom < 120
  }, [])

  useLayoutEffect(() => {
    if (pinnedToBottom.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  const scopedDocuments = channelDocuments.filter((d) => scopedDocumentIds.includes(d.id))

  const handleSend = (text) => {
    const sent = sendMessage(text, scopedDocumentIds)
    if (sent) setScopedDocumentIds([])
    return sent
  }

  const handleAskDocument = (doc) => {
    setScopedDocumentIds([doc.id])
  }

  if (!loaded) {
    return (
      <div className="boot">
        <span className="spinner spinner-large" />
        <p>Connecting to Omnivores...</p>
      </div>
    )
  }

  return (
    <div className="app">
      <Sidebar
        channels={channels}
        activeChannelId={activeChannelId}
        onSelectChannel={selectChannel}
        agents={channelAgents}
        documents={channelDocuments}
        uploads={uploads}
        limits={limits}
        onUpload={uploadDocument}
        onDeleteDocument={deleteDocument}
        onAskDocument={handleAskDocument}
      />

      <main className="chat">
        <Header
          channel={activeChannel}
          connection={connection}
          config={config}
          typingNames={typingNames}
        />

        <div className="messages" ref={scrollRef} onScroll={handleScroll}>
          {messages.length === 0 ? (
            <EmptyChannel
              agents={channelAgents}
              channelName={activeChannel?.name || activeChannelId}
            />
          ) : (
            messages.map((message) => (
              <Message
                key={message.id}
                message={message}
                agent={channelAgents.find((a) => a.id === message.author?.id)}
                onCancel={message.status === 'streaming' ? cancelMessage : undefined}
              />
            ))
          )}
        </div>

        <Composer
          agents={channelAgents}
          scopedDocuments={scopedDocuments}
          onClearScope={() => setScopedDocumentIds([])}
          onSend={handleSend}
          onTyping={setTyping}
          disabled={!isConnected}
          disabledReason="Reconnecting to the server..."
        />
      </main>

      <Toasts toasts={toasts} onDismiss={dismissToast} />
    </div>
  )
}
