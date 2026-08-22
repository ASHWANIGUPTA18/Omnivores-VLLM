/**
 * Channel list, agent roster, and the document panel.
 *
 * The document panel is where the RAG pipeline becomes visible: a file moves
 * through uploading -> processing -> indexing -> ready with a real progress
 * bar driven by backend events, and a failed document stays on screen with the
 * reason attached rather than disappearing.
 */

import { useRef, useState } from 'react'

function ChannelList({ channels, activeId, onSelect }) {
  return (
    <nav className="channel-list">
      <div className="sidebar-heading">Channels</div>
      {channels.map((channel) => (
        <button
          key={channel.id}
          type="button"
          className={`channel ${channel.id === activeId ? 'channel-active' : ''}`}
          onClick={() => onSelect(channel.id)}
          title={channel.description}
        >
          <span className="channel-hash">#</span>
          <span className="channel-name">{channel.name}</span>
        </button>
      ))}
    </nav>
  )
}

function AgentList({ agents }) {
  if (!agents.length) return null
  return (
    <div className="agent-list">
      <div className="sidebar-heading">Agents here</div>
      {agents.map((agent) => (
        <div key={agent.id} className="agent-row" title={agent.description}>
          <span className="agent-avatar" style={{ background: agent.color }}>
            {agent.avatar}
          </span>
          <div className="agent-info">
            <span className="agent-name">@{agent.id}</span>
            <span className="agent-tools">
              {agent.tools.length ? agent.tools.join(', ') : 'no tools'}
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}

const STAGE_LABELS = {
  uploading: 'Uploading',
  stored: 'Stored',
  parsed: 'Reading',
  chunked: 'Splitting',
  embedding: 'Embedding',
  embedded: 'Embedded',
  indexed: 'Indexed',
  reindexing: 'Re-indexing',
  failed: 'Failed',
}

function DocumentRow({ document, onDelete, onAsk }) {
  const ready = document.status === 'ready'
  const failed = document.status === 'failed'
  const percent = Math.round((document.progress || 0) * 100)

  return (
    <div className={`document document-${document.status}`}>
      <div className="document-head">
        <span className="document-icon">{failed ? '⚠' : ready ? '📄' : '⏳'}</span>
        <span className="document-name" title={document.filename}>
          {document.filename}
        </span>
      </div>

      {!ready && !failed && (
        <div className="document-progress">
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${percent}%` }} />
          </div>
          <span className="progress-label">
            {STAGE_LABELS[document.stage] || document.stage} · {percent}%
          </span>
        </div>
      )}

      {ready && (
        <div className="document-meta">
          {document.page_count ? `${document.page_count} pages · ` : ''}
          {document.chunk_count} chunks
        </div>
      )}

      {failed && <div className="document-error">{document.error}</div>}

      <div className="document-actions">
        {ready && (
          <button type="button" onClick={() => onAsk(document)}>
            Ask
          </button>
        )}
        <button type="button" className="danger" onClick={() => onDelete(document.id)}>
          Delete
        </button>
      </div>
    </div>
  )
}

function DocumentPanel({ documents, uploads, limits, onUpload, onDelete, onAsk }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const accept = (limits?.allowed_extensions || ['.pdf', '.txt', '.md', '.docx']).join(',')

  const handleFiles = (fileList) => {
    for (const file of Array.from(fileList || [])) onUpload(file)
  }

  return (
    <div className="document-panel">
      <div className="sidebar-heading">Documents</div>

      <div
        className={`dropzone ${dragging ? 'dropzone-active' : ''}`}
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setDragging(false)
          handleFiles(event.dataTransfer.files)
        }}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={accept}
          hidden
          onChange={(event) => {
            handleFiles(event.target.files)
            event.target.value = ''
          }}
        />
        <span className="dropzone-icon">＋</span>
        <span>Drop files or click to upload</span>
        <span className="dropzone-hint">
          {accept} · up to {limits?.max_upload_size_mb ?? 50} MB
        </span>
      </div>

      {Object.entries(uploads).map(([id, upload]) => (
        <div key={id} className="document document-uploading">
          <div className="document-head">
            <span className="document-icon">⬆</span>
            <span className="document-name" title={upload.name}>
              {upload.name}
            </span>
          </div>
          <div className="document-progress">
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${Math.round(upload.progress * 100)}%` }}
              />
            </div>
            <span className="progress-label">
              Uploading · {Math.round(upload.progress * 100)}%
            </span>
          </div>
        </div>
      ))}

      {documents.length === 0 && Object.keys(uploads).length === 0 && (
        <p className="empty-hint">
          Upload a PDF, Word file, Markdown or text file, then ask an agent about it.
        </p>
      )}

      {documents.map((document) => (
        <DocumentRow
          key={document.id}
          document={document}
          onDelete={onDelete}
          onAsk={onAsk}
        />
      ))}
    </div>
  )
}

export function Sidebar({
  channels,
  activeChannelId,
  onSelectChannel,
  agents,
  documents,
  uploads,
  limits,
  onUpload,
  onDeleteDocument,
  onAskDocument,
}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">◆</span>
        <span className="brand-name">Omnivores</span>
      </div>

      <ChannelList
        channels={channels}
        activeId={activeChannelId}
        onSelect={onSelectChannel}
      />
      <AgentList agents={agents} />
      <DocumentPanel
        documents={documents}
        uploads={uploads}
        limits={limits}
        onUpload={onUpload}
        onDelete={onDeleteDocument}
        onAsk={onAskDocument}
      />
    </aside>
  )
}
