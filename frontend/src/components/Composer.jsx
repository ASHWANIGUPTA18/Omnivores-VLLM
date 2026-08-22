/**
 * The message composer.
 *
 * Carries the @mention autocomplete, because mentions are the routing
 * mechanism - a user who cannot discover the agent names cannot use the
 * product. It also surfaces the document scope, so "ask this document" is
 * visible and revocable rather than an invisible mode.
 */

import { useEffect, useMemo, useRef, useState } from 'react'

const TYPING_IDLE_MS = 1500

export function Composer({
  agents,
  scopedDocuments,
  onClearScope,
  onSend,
  onTyping,
  disabled,
  disabledReason,
}) {
  const [value, setValue] = useState('')
  const [mentionQuery, setMentionQuery] = useState(null)
  const [highlight, setHighlight] = useState(0)
  const textareaRef = useRef(null)
  const typingTimer = useRef(null)

  // Grow the textarea to fit, up to a cap.
  useEffect(() => {
    const node = textareaRef.current
    if (!node) return
    node.style.height = 'auto'
    node.style.height = `${Math.min(node.scrollHeight, 200)}px`
  }, [value])

  const suggestions = useMemo(() => {
    if (mentionQuery === null) return []
    const query = mentionQuery.toLowerCase()
    return agents.filter((a) => a.id.startsWith(query) || a.display_name.toLowerCase().includes(query))
  }, [agents, mentionQuery])

  const updateMentionState = (text, caret) => {
    // Only the token immediately before the caret can be an active mention.
    const before = text.slice(0, caret)
    const match = before.match(/(?:^|\s)@([\w-]*)$/)
    setMentionQuery(match ? match[1] : null)
    setHighlight(0)
  }

  const handleChange = (event) => {
    const text = event.target.value
    setValue(text)
    updateMentionState(text, event.target.selectionStart)

    onTyping?.(true)
    clearTimeout(typingTimer.current)
    typingTimer.current = setTimeout(() => onTyping?.(false), TYPING_IDLE_MS)
  }

  const applySuggestion = (agent) => {
    const node = textareaRef.current
    const caret = node?.selectionStart ?? value.length
    const before = value.slice(0, caret).replace(/@[\w-]*$/, `@${agent.id} `)
    const next = before + value.slice(caret)
    setValue(next)
    setMentionQuery(null)
    requestAnimationFrame(() => {
      node?.focus()
      node?.setSelectionRange(before.length, before.length)
    })
  }

  const submit = () => {
    const text = value.trim()
    if (!text || disabled) return
    if (onSend(text)) {
      setValue('')
      setMentionQuery(null)
      onTyping?.(false)
      clearTimeout(typingTimer.current)
    }
  }

  const handleKeyDown = (event) => {
    if (suggestions.length > 0) {
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        setHighlight((h) => (h + 1) % suggestions.length)
        return
      }
      if (event.key === 'ArrowUp') {
        event.preventDefault()
        setHighlight((h) => (h - 1 + suggestions.length) % suggestions.length)
        return
      }
      if (event.key === 'Tab' || (event.key === 'Enter' && !event.shiftKey)) {
        event.preventDefault()
        applySuggestion(suggestions[highlight])
        return
      }
      if (event.key === 'Escape') {
        setMentionQuery(null)
        return
      }
    }

    // Enter sends; Shift+Enter inserts a newline.
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <div className="composer">
      {scopedDocuments.length > 0 && (
        <div className="scope-bar">
          <span className="scope-label">Asking about</span>
          {scopedDocuments.map((document) => (
            <span key={document.id} className="scope-chip">
              📄 {document.filename}
            </span>
          ))}
          <button type="button" className="scope-clear" onClick={onClearScope}>
            clear
          </button>
        </div>
      )}

      {suggestions.length > 0 && (
        <div className="mention-menu">
          {suggestions.map((agent, index) => (
            <button
              key={agent.id}
              type="button"
              className={`mention-option ${index === highlight ? 'mention-active' : ''}`}
              onMouseEnter={() => setHighlight(index)}
              onClick={() => applySuggestion(agent)}
            >
              <span className="agent-avatar" style={{ background: agent.color }}>
                {agent.avatar}
              </span>
              <span className="mention-id">@{agent.id}</span>
              <span className="mention-description">{agent.description}</span>
            </button>
          ))}
        </div>
      )}

      <div className="composer-input">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onBlur={() => onTyping?.(false)}
          rows={1}
          placeholder={
            disabled
              ? disabledReason || 'Disconnected'
              : 'Message the channel, or @mention an agent...'
          }
          disabled={disabled}
        />
        <button type="button" className="send-button" onClick={submit} disabled={disabled || !value.trim()}>
          Send
        </button>
      </div>

      <div className="composer-hint">
        <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line ·{' '}
        <kbd>@</kbd> to mention an agent
      </div>
    </div>
  )
}
