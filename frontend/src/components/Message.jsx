/**
 * One message in the timeline.
 *
 * An agent message is more than text: it carries the tools it ran, the sources
 * it cited, and the metrics of the run. Showing the tool strip while the tool
 * is still running is what makes a multi-second agent turn feel like progress
 * rather than a hang.
 */

import { useRef, useState } from 'react'
import { Markdown } from './Markdown'

function timestamp(iso) {
  if (!iso) return ''
  const date = new Date(iso)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const TOOL_LABELS = {
  search_knowledge_base: 'Searching documents',
  web_search: 'Searching the web',
  run_python: 'Running Python',
}

function ToolStrip({ invocations }) {
  if (!invocations?.length) return null

  return (
    <div className="tool-strip">
      {invocations.map((invocation) => {
        const label = TOOL_LABELS[invocation.name] || invocation.name
        return (
          <div key={invocation.id} className={`tool-chip tool-${invocation.status}`}>
            <span className="tool-icon">
              {invocation.status === 'running' && <span className="spinner" />}
              {invocation.status === 'completed' && '✓'}
              {invocation.status === 'failed' && '✕'}
            </span>
            <span className="tool-label">{label}</span>
            {invocation.arguments?.query && (
              <span className="tool-arg">“{invocation.arguments.query}”</span>
            )}
            {invocation.result_preview && invocation.status !== 'running' && (
              <span className="tool-result">{invocation.result_preview}</span>
            )}
            {invocation.duration_ms != null && (
              <span className="tool-duration">{invocation.duration_ms}ms</span>
            )}
          </div>
        )
      })}
    </div>
  )
}

function Citations({ citations, refs }) {
  const [expanded, setExpanded] = useState(null)
  if (!citations?.length) return null

  return (
    <div className="citations">
      <div className="citations-label">Sources</div>
      <div className="citations-list">
        {citations.map((citation, index) => {
          const number = index + 1
          const isOpen = expanded === number
          return (
            <div
              key={citation.chunk_id || number}
              className="citation"
              ref={(node) => {
                if (refs) refs.current[number] = node
              }}
            >
              <button
                type="button"
                className="citation-head"
                onClick={() => setExpanded(isOpen ? null : number)}
              >
                <span className="citation-number">{number}</span>
                <span className="citation-file">{citation.filename}</span>
                {citation.page != null && (
                  <span className="citation-page">page {citation.page}</span>
                )}
                {citation.score != null && (
                  <span className="citation-score">{Math.round(citation.score * 100)}%</span>
                )}
              </button>
              {isOpen && <div className="citation-snippet">{citation.snippet}</div>}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Metrics({ metrics }) {
  const [open, setOpen] = useState(false)
  if (!metrics) return null

  const parts = []
  if (metrics.ttft_ms != null) parts.push(`${Math.round(metrics.ttft_ms)}ms to first token`)
  if (metrics.tokens_per_second) parts.push(`${metrics.tokens_per_second} tok/s`)
  if (metrics.output_tokens) parts.push(`${metrics.output_tokens} tokens out`)
  if (metrics.total_ms) parts.push(`${(metrics.total_ms / 1000).toFixed(1)}s total`)
  if (!parts.length) return null

  return (
    <div className="metrics">
      <button type="button" className="metrics-toggle" onClick={() => setOpen(!open)}>
        {open ? '▾' : '▸'} {parts.join(' · ')}
      </button>
      {open && (
        <dl className="metrics-detail">
          <div><dt>Rounds</dt><dd>{metrics.rounds}</dd></div>
          <div><dt>Input tokens</dt><dd>{metrics.input_tokens}</dd></div>
          <div><dt>Output tokens</dt><dd>{metrics.output_tokens}</dd></div>
          <div><dt>Context tokens</dt><dd>{metrics.context_tokens}</dd></div>
          {metrics.retrieved_chunks > 0 && (
            <div><dt>Chunks retrieved</dt><dd>{metrics.retrieved_chunks}</dd></div>
          )}
          {metrics.retrieval_ms > 0 && (
            <div><dt>Retrieval</dt><dd>{Math.round(metrics.retrieval_ms)}ms</dd></div>
          )}
        </dl>
      )}
    </div>
  )
}

export function Message({ message, agent, onCancel }) {
  const citationRefs = useRef({})
  const { author } = message
  const isAgent = author?.type === 'agent'
  const isSystem = author?.type === 'system'
  const isStreaming = message.status === 'streaming'
  const isFailed = message.status === 'failed'

  const scrollToCitation = (number) => {
    citationRefs.current[number]?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    citationRefs.current[number]?.classList.add('citation-flash')
    setTimeout(() => citationRefs.current[number]?.classList.remove('citation-flash'), 1200)
  }

  if (isSystem) {
    return (
      <div className="message message-system">
        <span className="system-icon">ⓘ</span>
        <span>{message.content}</span>
      </div>
    )
  }

  return (
    <article className={`message ${isAgent ? 'message-agent' : 'message-user'}`}>
      <div
        className="avatar"
        style={isAgent && agent ? { background: agent.color } : undefined}
      >
        {isAgent ? author.avatar || '🤖' : author.display_name?.[0]?.toUpperCase() || 'U'}
      </div>

      <div className="message-body">
        <header className="message-header">
          <span className="message-author">{author?.display_name}</span>
          {isAgent && <span className="agent-tag">agent</span>}
          <time className="message-time">{timestamp(message.created_at)}</time>
          {isStreaming && (
            <>
              <span className="streaming-badge">generating</span>
              {onCancel && (
                <button
                  type="button"
                  className="cancel-button"
                  onClick={() => onCancel(message.id)}
                >
                  stop
                </button>
              )}
            </>
          )}
        </header>

        <ToolStrip invocations={message.tool_invocations} />

        {message.content ? (
          <Markdown text={message.content} onCitationClick={scrollToCitation} />
        ) : (
          isStreaming && <div className="thinking">thinking<span className="dots" /></div>
        )}

        {isFailed && message.error && (
          <div className="message-error">
            <strong>{message.error.code}</strong>
            <span>{message.error.message}</span>
          </div>
        )}

        <Citations citations={message.citations} refs={citationRefs} />
        <Metrics metrics={message.metrics} />
      </div>
    </article>
  )
}
