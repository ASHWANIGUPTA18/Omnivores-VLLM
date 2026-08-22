/**
 * A small Markdown renderer.
 *
 * Written by hand rather than pulled from npm for one reason that matters:
 * it renders to React elements and never touches `dangerouslySetInnerHTML`.
 * Model output is untrusted text - it can contain anything a document or a web
 * page contained - so the safest renderer is one that structurally cannot
 * inject HTML.
 *
 * Supports what agents actually emit: fenced code, inline code, bold, italic,
 * links, headings, bullet and numbered lists, blockquotes, and `[1]`-style
 * citation markers which become clickable chips.
 */

import { useMemo } from 'react'

const INLINE_PATTERN =
  /(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\n]+\*)|(\[[^\]]+\]\([^)\s]+\))|(\[\d+\])/g

function renderInline(text, { onCitationClick } = {}) {
  const nodes = []
  let cursor = 0
  let key = 0

  for (const match of text.matchAll(INLINE_PATTERN)) {
    if (match.index > cursor) {
      nodes.push(text.slice(cursor, match.index))
    }
    const token = match[0]

    if (token.startsWith('`')) {
      nodes.push(<code key={key++} className="md-code">{token.slice(1, -1)}</code>)
    } else if (token.startsWith('**')) {
      nodes.push(<strong key={key++}>{token.slice(2, -2)}</strong>)
    } else if (token.startsWith('[') && token.includes('](')) {
      const label = token.slice(1, token.indexOf(']'))
      const href = token.slice(token.indexOf('](') + 2, -1)
      const safe = /^https?:\/\//i.test(href)
      nodes.push(
        safe ? (
          <a key={key++} href={href} target="_blank" rel="noopener noreferrer">
            {label}
          </a>
        ) : (
          // A non-http scheme (javascript:, data:) is rendered as plain text.
          <span key={key++}>{label}</span>
        )
      )
    } else if (/^\[\d+\]$/.test(token)) {
      const index = Number(token.slice(1, -1))
      nodes.push(
        <button
          key={key++}
          type="button"
          className="citation-marker"
          onClick={() => onCitationClick?.(index)}
          title={`Jump to source ${index}`}
        >
          {index}
        </button>
      )
    } else {
      nodes.push(<em key={key++}>{token.slice(1, -1)}</em>)
    }

    cursor = match.index + token.length
  }

  if (cursor < text.length) nodes.push(text.slice(cursor))
  return nodes
}

/** Split source into blocks, keeping fenced code intact. */
function parseBlocks(source) {
  const blocks = []
  const lines = source.split('\n')
  let index = 0

  while (index < lines.length) {
    const line = lines[index]

    // Fenced code block.
    if (line.trimStart().startsWith('```')) {
      const language = line.trim().slice(3).trim()
      const body = []
      index += 1
      while (index < lines.length && !lines[index].trimStart().startsWith('```')) {
        body.push(lines[index])
        index += 1
      }
      index += 1 // closing fence (absent if the stream was cut off)
      blocks.push({ type: 'code', language, content: body.join('\n') })
      continue
    }

    if (!line.trim()) {
      index += 1
      continue
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/)
    if (heading) {
      blocks.push({ type: 'heading', level: heading[1].length, content: heading[2] })
      index += 1
      continue
    }

    if (/^\s*>\s?/.test(line)) {
      const body = []
      while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
        body.push(lines[index].replace(/^\s*>\s?/, ''))
        index += 1
      }
      blocks.push({ type: 'quote', content: body.join('\n') })
      continue
    }

    const bullet = /^\s*[-*+]\s+/
    const numbered = /^\s*\d+\.\s+/
    if (bullet.test(line) || numbered.test(line)) {
      const ordered = numbered.test(line)
      const pattern = ordered ? numbered : bullet
      const items = []
      while (index < lines.length && pattern.test(lines[index])) {
        items.push(lines[index].replace(pattern, ''))
        index += 1
      }
      blocks.push({ type: 'list', ordered, items })
      continue
    }

    // Paragraph: consume until a blank line or the start of another block.
    const paragraph = []
    while (
      index < lines.length &&
      lines[index].trim() &&
      !lines[index].trimStart().startsWith('```') &&
      !/^(#{1,4})\s+/.test(lines[index]) &&
      !bullet.test(lines[index]) &&
      !numbered.test(lines[index]) &&
      !/^\s*>\s?/.test(lines[index])
    ) {
      paragraph.push(lines[index])
      index += 1
    }
    blocks.push({ type: 'paragraph', content: paragraph.join('\n') })
  }

  return blocks
}

function CodeBlock({ language, content }) {
  const copy = () => navigator.clipboard?.writeText(content)
  return (
    <div className="md-codeblock">
      <div className="md-codeblock-header">
        <span>{language || 'text'}</span>
        <button type="button" onClick={copy} title="Copy">
          copy
        </button>
      </div>
      <pre>
        <code>{content}</code>
      </pre>
    </div>
  )
}

export function Markdown({ text, onCitationClick }) {
  const blocks = useMemo(() => parseBlocks(text || ''), [text])

  return (
    <div className="markdown">
      {blocks.map((block, index) => {
        switch (block.type) {
          case 'code':
            return <CodeBlock key={index} language={block.language} content={block.content} />

          case 'heading': {
            const Tag = `h${Math.min(block.level + 2, 6)}`
            return (
              <Tag key={index} className="md-heading">
                {renderInline(block.content, { onCitationClick })}
              </Tag>
            )
          }

          case 'quote':
            return (
              <blockquote key={index}>
                {renderInline(block.content, { onCitationClick })}
              </blockquote>
            )

          case 'list': {
            const Tag = block.ordered ? 'ol' : 'ul'
            return (
              <Tag key={index}>
                {block.items.map((item, n) => (
                  <li key={n}>{renderInline(item, { onCitationClick })}</li>
                ))}
              </Tag>
            )
          }

          default:
            return <p key={index}>{renderInline(block.content, { onCitationClick })}</p>
        }
      })}
    </div>
  )
}
