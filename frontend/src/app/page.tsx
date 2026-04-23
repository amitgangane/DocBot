'use client'

import { useEffect, useRef, useState } from 'react'
import {
  Bot,
  ChevronDown,
  ChevronUp,
  Check,
  CircleUserRound,
  Clock3,
  Copy,
  History,
  Loader2,
  Menu,
  PlusCircle,
  Search,
  Send,
  Settings2,
  Sparkles,
  Trash2,
  Upload,
  MoreVertical,
  X,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: number
  sourceItems?: SourceItem[]
  statusTrail?: string[]
  activeStatus?: string
  kind?: 'default' | 'error'
}

interface Thread {
  id: string
  title: string
  createdAt: string
}

interface UploadRecord {
  id: string
  filename: string
  documentId?: string
  status: 'success' | 'error'
  chunkCount?: number
  createdAt: string
  detail: string
}

interface IndexedDocument {
  document_id: string
  filename: string
  source_path: string
  chunk_count: number
  page_count?: number | null
}

interface SourceItem {
  document_id: string
  filename: string
  source_path: string
  page_number?: number | null
  chunk_id?: string | null
  excerpt: string
}

const createThreadId = () => `thread-${Date.now()}`
const formatRelativeTime = (value: string) => {
  const timestamp = new Date(value).getTime()
  const diffMs = Date.now() - timestamp
  const diffMinutes = Math.max(1, Math.floor(diffMs / 60000))

  if (diffMinutes < 60) return `${diffMinutes}m ago`

  const diffHours = Math.floor(diffMinutes / 60)
  if (diffHours < 24) return `${diffHours}h ago`

  const diffDays = Math.floor(diffHours / 24)
  if (diffDays < 7) return `${diffDays}d ago`

  return new Date(value).toLocaleDateString()
}

const getThreadTitle = (threads: Thread[], currentThreadId: string) => {
  return threads.find((thread) => thread.id === currentThreadId)?.title || 'New conversation'
}

const normalizeMathDelimiters = (content: string) => {
  return content
    .replace(/\\\[((?:.|\n)+?)\\\]/g, (_, expr: string) => `$$${expr.trim()}$$`)
    .replace(/\\\(((?:.|\n)+?)\\\)/g, (_, expr: string) => `$${expr.trim()}$`)
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isLoadingThread, setIsLoadingThread] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [threads, setThreads] = useState<Thread[]>([])
  const [currentThreadId, setCurrentThreadId] = useState<string>('')
  const [uploadStatus, setUploadStatus] = useState<{ kind: 'idle' | 'progress' | 'success' | 'error'; message: string } | null>(null)
  const [recentUploads, setRecentUploads] = useState<UploadRecord[]>([])
  const [indexedDocuments, setIndexedDocuments] = useState<IndexedDocument[]>([])
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false)
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null)
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({})
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [uploadPanelOpen, setUploadPanelOpen] = useState(false)
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'

  useEffect(() => {
    const threadId = createThreadId()
    setCurrentThreadId(threadId)

    const savedThreads = localStorage.getItem('docbot-threads')
    if (savedThreads) {
      setThreads(JSON.parse(savedThreads))
    }

    const savedUploads = localStorage.getItem('docbot-uploads')
    if (savedUploads) {
      setRecentUploads(JSON.parse(savedUploads))
    }
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return

    textarea.style.height = '0px'
    const nextHeight = Math.min(textarea.scrollHeight, 160)
    textarea.style.height = `${Math.max(nextHeight, 40)}px`
  }, [input])

  useEffect(() => {
    if (!uploadPanelOpen) return
    void loadIndexedDocuments()
  }, [uploadPanelOpen])

  const persistThreads = (nextThreads: Thread[]) => {
    setThreads(nextThreads)
    localStorage.setItem('docbot-threads', JSON.stringify(nextThreads))
  }

  const persistUploads = (nextUploads: UploadRecord[]) => {
    setRecentUploads(nextUploads)
    localStorage.setItem('docbot-uploads', JSON.stringify(nextUploads))
  }

  const saveThread = (threadId: string, firstMessage: string) => {
    const newThread: Thread = {
      id: threadId,
      title: firstMessage.slice(0, 50) + (firstMessage.length > 50 ? '...' : ''),
      createdAt: new Date().toISOString(),
    }
    const nextThreads = [newThread, ...threads.filter((thread) => thread.id !== threadId)]
    persistThreads(nextThreads)
  }

  const saveUploadRecord = (record: UploadRecord) => {
    const nextUploads = [record, ...recentUploads].slice(0, 6)
    persistUploads(nextUploads)
  }

  const toggleSources = (messageId: string) => {
    setExpandedSources((prev) => ({
      ...prev,
      [messageId]: !prev[messageId],
    }))
  }

  const reconcileRecentUploads = (documents: IndexedDocument[]) => {
    setRecentUploads((prev) => {
      const nextUploads = prev.filter((upload) => {
        if (upload.status !== 'success') return true

        return documents.some((document) => {
          if (upload.documentId) {
            return document.document_id === upload.documentId
          }
          return document.filename === upload.filename
        })
      })

      localStorage.setItem('docbot-uploads', JSON.stringify(nextUploads))
      return nextUploads
    })
  }

  const loadIndexedDocuments = async () => {
    setIsLoadingDocuments(true)

    try {
      const response = await fetch(`${API_URL}/documents`)
      if (!response.ok) {
        throw new Error('Failed to load indexed documents')
      }

      const data = await response.json()
      const documents = Array.isArray(data) ? data : []
      setIndexedDocuments(documents)
      reconcileRecentUploads(documents)
    } catch (error) {
      setUploadStatus({
        kind: 'error',
        message: error instanceof Error ? error.message : 'Failed to load indexed documents.',
      })
    } finally {
      setIsLoadingDocuments(false)
    }
  }

  const startNewChat = () => {
    const newThreadId = createThreadId()
    setCurrentThreadId(newThreadId)
    setMessages([])
    setSidebarOpen(false)
  }

  const deleteThread = (threadId: string) => {
    const nextThreads = threads.filter((thread) => thread.id !== threadId)
    persistThreads(nextThreads)
    if (currentThreadId === threadId) {
      startNewChat()
    }
  }

  const handleCopyMessage = async (messageId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content)
      setCopiedMessageId(messageId)
      window.setTimeout(() => setCopiedMessageId(null), 1800)
    } catch {
      setCopiedMessageId(null)
    }
  }

  const loadThread = async (threadId: string) => {
    setCurrentThreadId(threadId)
    setIsLoadingThread(true)
    setMessages([])
    setSidebarOpen(false)

    const controller = new AbortController()
    const timeoutId = window.setTimeout(() => controller.abort(), 8000)

    try {
      const response = await fetch(`${API_URL}/threads/${threadId}/history`, {
        signal: controller.signal,
      })
      if (!response.ok) {
        throw new Error('Failed to load conversation')
      }

      const data = await response.json()
      if (data.messages && data.messages.length > 0) {
        const loadedMessages: Message[] = data.messages.map((msg: { role: 'user' | 'assistant'; content: string; source_items?: SourceItem[] }, index: number) => ({
          id: `${threadId}-${index}`,
          role: msg.role,
          content: msg.content,
          sourceItems: msg.source_items || [],
        }))
        setMessages(loadedMessages)
      }
    } catch (error) {
      setMessages([
        {
          id: `error-${threadId}`,
          role: 'assistant',
          content:
            error instanceof Error && error.name === 'AbortError'
              ? 'Loading this conversation timed out. Please make sure the backend is running and try again.'
              : error instanceof Error
                ? error.message
                : 'Failed to load conversation.',
          kind: 'error',
        },
      ])
    } finally {
      window.clearTimeout(timeoutId)
      setIsLoadingThread(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading || !currentThreadId) return

    const question = input.trim()
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: question,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setIsLoading(true)
    setSidebarOpen(false)

    if (messages.length === 0) {
      saveThread(currentThreadId, userMessage.content)
    }

    const assistantMessageId = `${Date.now() + 1}`
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      sources: 0,
      statusTrail: [],
      activeStatus: 'Starting the RAG pipeline',
      kind: 'default',
    }
    setMessages((prev) => [...prev, assistantMessage])

    try {
      const response = await fetch(`${API_URL}/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
        },
        body: JSON.stringify({
          question,
          thread_id: currentThreadId,
        }),
      })

      if (!response.ok) throw new Error('Failed to get response')
      if (!response.body) throw new Error('Streaming response body is missing')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let fullContent = ''
      let sources = 0
      let buffer = ''

      const applyAssistantUpdate = (updates: Partial<Message>) => {
        setMessages((prev) =>
          prev.map((msg) => (msg.id === assistantMessageId ? { ...msg, ...updates } : msg))
        )
      }

      const processEvent = (eventBlock: string) => {
        const lines = eventBlock.split('\n')
        let eventType = ''
        const dataLines: string[] = []

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim()
          } else if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trim())
          }
        }

        if (!eventType || dataLines.length === 0) return false

        const parsed = JSON.parse(dataLines.join('\n'))

        if (eventType === 'status') {
          setMessages((prev) =>
            prev.map((msg) => {
              if (msg.id !== assistantMessageId) return msg

              const nextTrail = msg.statusTrail?.includes(parsed.message)
                ? msg.statusTrail
                : [...(msg.statusTrail || []), parsed.message]

              return {
                ...msg,
                statusTrail: nextTrail,
                activeStatus: parsed.message,
              }
            })
          )
          return false
        }

        if (eventType === 'metadata') {
          sources = parsed.sources || 0
          applyAssistantUpdate({ sources, sourceItems: parsed.source_items || [] })
          return false
        }

        if (eventType === 'token') {
          fullContent += parsed.token || ''
          applyAssistantUpdate({ content: fullContent, sources, activeStatus: undefined })
          return false
        }

        if (eventType === 'error') {
          throw new Error(parsed.message || 'Streaming request failed')
        }

        if (eventType === 'done') {
          applyAssistantUpdate({ content: fullContent, sources, activeStatus: undefined })
          return true
        }

        return false
      }

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        while (buffer.includes('\n\n')) {
          const separatorIndex = buffer.indexOf('\n\n')
          const eventBlock = buffer.slice(0, separatorIndex)
          buffer = buffer.slice(separatorIndex + 2)

          if (!eventBlock.trim()) continue

          const isDone = processEvent(eventBlock)
          if (isDone) {
            buffer = ''
            break
          }
        }
      }

      const finalChunk = decoder.decode()
      if (finalChunk) buffer += finalChunk
      if (buffer.trim()) processEvent(buffer.trim())
    } catch (error) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: error instanceof Error ? error.message : 'Sorry, I encountered an error. Please make sure the backend is running.',
                kind: 'error',
                activeStatus: undefined,
              }
            : msg
        )
      )
    } finally {
      setIsLoading(false)
    }
  }

  const handleComposerKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!input.trim() || isLoading || !currentThreadId) return

      void handleSubmit(e)
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !file.name.endsWith('.pdf')) {
      setUploadStatus({ kind: 'error', message: 'Please select a PDF file.' })
      return
    }

    setIsUploading(true)
    setUploadPanelOpen(true)
    setUploadStatus({ kind: 'progress', message: `Uploading ${file.name} and indexing your document...` })

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_URL}/ingest/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('Upload failed')

      const data = await response.json()
      const successMessage = `${file.name} uploaded successfully. ${data.chunk_count} chunks were indexed.`
      setUploadStatus({ kind: 'success', message: successMessage })
      saveUploadRecord({
        id: `${Date.now()}`,
        filename: file.name,
        documentId: data.document_id,
        status: 'success',
        chunkCount: data.chunk_count,
        createdAt: new Date().toISOString(),
        detail: successMessage,
      })
      await loadIndexedDocuments()

      window.setTimeout(() => setUploadStatus(null), 5000)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Upload failed. Is the backend running?'
      setUploadStatus({ kind: 'error', message })
      saveUploadRecord({
        id: `${Date.now()}`,
        filename: file.name,
        status: 'error',
        createdAt: new Date().toISOString(),
        detail: message,
      })
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleDeleteDocument = async (document: IndexedDocument) => {
    const shouldDelete = window.confirm(`Delete ${document.filename} from the index?`)
    if (!shouldDelete) return

    setDeletingDocumentId(document.document_id)

    try {
      const response = await fetch(`${API_URL}/documents/${document.document_id}`, {
        method: 'DELETE',
      })

      if (!response.ok) {
        throw new Error('Failed to delete document')
      }

      setIndexedDocuments((prev) => prev.filter((item) => item.document_id !== document.document_id))
      setRecentUploads((prev) => {
        const nextUploads = prev.filter((upload) =>
          upload.documentId
            ? upload.documentId !== document.document_id
            : upload.filename !== document.filename
        )
        localStorage.setItem('docbot-uploads', JSON.stringify(nextUploads))
        return nextUploads
      })
      setUploadStatus({
        kind: 'success',
        message: `${document.filename} was removed from the index.`,
      })
    } catch (error) {
      setUploadStatus({
        kind: 'error',
        message: error instanceof Error ? error.message : 'Failed to delete document.',
      })
    } finally {
      setDeletingDocumentId(null)
    }
  }

  const currentThreadTitle = getThreadTitle(threads, currentThreadId)

  return (
    <div className="relative flex h-screen overflow-hidden bg-[#f7f9fb] text-slate-900">
      {sidebarOpen ? (
        <button
          aria-label="Close sidebar"
          className="fixed inset-0 z-40 bg-slate-900/20 lg:hidden"
          onClick={() => setSidebarOpen(false)}
          type="button"
        />
      ) : null}

      <div className="relative flex h-full min-h-0 w-full">
        <aside
          className={`fixed inset-y-0 left-0 z-50 flex w-[280px] max-w-[88vw] flex-col border-r border-slate-200 bg-slate-50 px-4 py-6 shadow-xl transition-transform duration-300 lg:static lg:translate-x-0 ${
            sidebarOpen ? 'translate-x-0' : '-translate-x-full'
          }`}
        >
          <div className="mb-8 flex items-center gap-3 px-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-800 text-white">
              <Bot size={18} />
            </div>
            <div>
              <h1 className="font-['Manrope','Avenir_Next','Segoe_UI',sans-serif] text-lg font-bold leading-none tracking-tight text-slate-900">
                DocBot
              </h1>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto">
            <button
              className="flex w-full items-center gap-3 border-l-2 border-slate-700 bg-white px-3 py-2.5 text-left text-sm font-medium tracking-tight text-slate-900 transition hover:bg-slate-100"
              onClick={startNewChat}
              type="button"
            >
              <PlusCircle size={16} />
              New chat
            </button>

            <div className="px-3 pb-2 pt-8">
              <span className="text-[11px] font-bold uppercase tracking-[0.24em] text-slate-400">Recents</span>
            </div>

            {threads.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-200 px-4 py-4 text-sm text-slate-400">
                Your conversations will appear here once you start chatting.
              </div>
            ) : (
              <div className="space-y-0.5">
                {threads.map((thread) => (
                  <button
                    className={`group flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition ${
                      currentThreadId === thread.id
                        ? 'bg-white text-slate-900'
                        : 'text-slate-500 hover:bg-slate-100 hover:text-slate-900'
                    }`}
                    key={thread.id}
                    onClick={() => loadThread(thread.id)}
                    type="button"
                  >
                    <History size={15} className="shrink-0" />
                    <div className="min-w-0 flex-1">
                      <span className="block truncate">{thread.title}</span>
                    </div>
                    <button
                      aria-label={`Delete ${thread.title}`}
                      className="rounded-md p-1 text-slate-400 opacity-0 transition hover:bg-red-50 hover:text-red-500 group-hover:opacity-100"
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteThread(thread.id)
                      }}
                      type="button"
                    >
                      <Trash2 size={13} />
                    </button>
                  </button>
                ))}
              </div>
            )}
          </div>
        </aside>

        <main className="flex min-w-0 flex-1 flex-col lg:ml-0">
          <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-slate-100 bg-white/80 px-4 shadow-sm backdrop-blur-md md:px-6">
            <div className="flex items-center gap-3">
              <button
                aria-label="Open sidebar"
                className="rounded-full p-2 text-slate-500 transition hover:bg-slate-100 lg:hidden"
                onClick={() => setSidebarOpen(true)}
                type="button"
              >
                <Menu size={18} />
              </button>
              <div className="flex items-center gap-2">
                <span className="max-w-[360px] truncate font-['Manrope','Avenir_Next','Segoe_UI',sans-serif] text-sm font-semibold tracking-tight text-slate-900 md:max-w-none">
                  {currentThreadTitle}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <button
                className="rounded-full p-2 text-slate-500 transition hover:bg-slate-100"
                type="button"
              >
                <CircleUserRound size={18} />
              </button>
              <button
                className="rounded-full p-2 text-slate-500 transition hover:bg-slate-100"
                type="button"
              >
                <MoreVertical size={18} />
              </button>
            </div>
          </header>

          <section className="flex-1 overflow-y-auto px-4 py-8 md:px-6">
            {isLoadingThread ? (
              <div className="flex h-full flex-col items-center justify-center text-slate-400">
                <Loader2 size={42} className="animate-spin text-slate-500" />
                <p className="mt-4 text-sm">Loading your conversation...</p>
              </div>
            ) : messages.length === 0 ? (
              <div className="mx-auto flex h-full w-full max-w-[800px] flex-col items-center justify-center px-6 text-center">
                <div className="rounded-[28px] border border-slate-200 bg-white px-10 py-12 shadow-sm">
                  <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-100 text-slate-800">
                    <Bot size={24} />
                  </div>
                  <h2 className="mt-6 font-['Manrope','Avenir_Next','Segoe_UI',sans-serif] text-[34px] font-bold tracking-[-0.02em] text-slate-900">
                    Ask anything from your PDFs
                  </h2>
                  <p className="mt-3 max-w-xl text-base text-slate-500">
                    Upload a document, ask a question, and get grounded answers with clear sources.
                  </p>
                </div>
              </div>
            ) : (
              <div className="mx-auto w-full max-w-[860px] space-y-8">
                {messages.map((message) => {
                  const isUser = message.role === 'user'
                  const isError = message.kind === 'error'

                  return (
                    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`} key={message.id}>
                      <div className={`min-w-0 ${isUser ? 'max-w-[80%]' : 'max-w-[90%]'}`}>
                        <div
                          className={`shadow-sm ${
                            isUser
                              ? 'rounded-2xl rounded-tr-none border border-slate-300 bg-slate-200 px-6 py-4 text-slate-900'
                              : isError
                                ? 'rounded-2xl rounded-tl-none border border-red-100 bg-red-50 p-6 text-red-700'
                                : 'rounded-2xl rounded-tl-none border border-slate-100 bg-white p-6 text-slate-900'
                          }`}
                        >
                          {isUser ? (
                            <p className="whitespace-pre-wrap text-base leading-7">{message.content}</p>
                          ) : (
                            <div className="space-y-4">
                              {!message.content && (message.statusTrail?.length || message.activeStatus) ? (
                                <div className="space-y-3">
                                  <div className="flex items-center gap-2 text-sm font-medium text-slate-400">
                                    <div className="flex items-center gap-1">
                                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-500" />
                                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-400/70 [animation-delay:120ms]" />
                                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-slate-300 [animation-delay:240ms]" />
                                    </div>
                                    <span className="italic">Assistant is thinking...</span>
                                  </div>
                                  <div className="space-y-2">
                                    {(message.statusTrail || []).map((status, index) => {
                                      const isActive = status === message.activeStatus
                                      return (
                                        <div
                                          className={`rounded-xl border px-3 py-2 text-sm transition ${
                                            isActive
                                              ? 'border-slate-300 bg-slate-100 text-slate-700'
                                              : 'border-slate-100 bg-slate-50 text-slate-500'
                                          }`}
                                          key={`${message.id}-status-${index}`}
                                        >
                                          {status}
                                        </div>
                                      )
                                    })}
                                  </div>
                                </div>
                              ) : null}

                              {message.content ? (
                                <div className="space-y-3">
                                  <div className="flex items-center justify-between gap-3">
                                    <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.24em] text-slate-400">
                                      <div className="flex h-6 w-6 items-center justify-center rounded bg-slate-100 text-slate-700">
                                        <Sparkles size={14} />
                                      </div>
                                      Answer
                                    </div>
                                    <button
                                      className="inline-flex items-center gap-1.5 rounded px-2 py-1 text-xs font-semibold text-slate-400 transition hover:bg-slate-50 hover:text-slate-600"
                                      onClick={() => handleCopyMessage(message.id, message.content)}
                                      type="button"
                                    >
                                      {copiedMessageId === message.id ? <Check size={14} /> : <Copy size={14} />}
                                      {copiedMessageId === message.id ? 'Copied' : 'Copy'}
                                    </button>
                                  </div>
                                  <div className="prose prose-slate max-w-none">
                                    <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                                      {normalizeMathDelimiters(message.content)}
                                    </ReactMarkdown>
                                  </div>
                                </div>
                              ) : !(message.statusTrail?.length || message.activeStatus) ? (
                                <Loader2 size={20} className="animate-spin text-slate-500" />
                              ) : null}
                            </div>
                          )}
                        </div>

                        {!isUser ? (
                          <div className="mt-2 flex flex-wrap items-center gap-3 px-2 text-xs text-slate-400">
                            {message.sources && message.sources > 0 ? (
                              <span className="rounded-full border border-slate-200 bg-white px-3 py-1">
                                Sources: {message.sources} chunks
                              </span>
                            ) : null}
                            {message.statusTrail && message.statusTrail.length > 0 && message.content ? (
                              <span className="rounded-full border border-slate-200 bg-white px-3 py-1">
                                Completed {message.statusTrail.length} retrieval steps
                              </span>
                            ) : null}
                          </div>
                        ) : null}

                        {!isUser && message.sourceItems && message.sourceItems.length > 0 ? (
                          <div className="mt-3 space-y-2 px-2">
                            <div className="flex items-center justify-between gap-3">
                              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
                                Sources
                              </p>
                              <button
                                className="inline-flex items-center gap-1 rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-medium text-slate-500 transition hover:border-slate-300 hover:bg-slate-50"
                                onClick={() => toggleSources(message.id)}
                                type="button"
                              >
                                {expandedSources[message.id] ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                                {expandedSources[message.id] ? 'Hide details' : 'Show details'}
                              </button>
                            </div>

                            <div className="flex flex-wrap gap-2">
                              {message.sourceItems.map((source) => (
                                <div
                                  className="inline-flex max-w-full items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-500"
                                  key={source.chunk_id || `${source.document_id}-${source.page_number}`}
                                >
                                  <span className="truncate font-medium text-slate-700">{source.filename}</span>
                                  {source.page_number ? (
                                    <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] text-slate-800">
                                      p.{source.page_number}
                                    </span>
                                  ) : null}
                                </div>
                              ))}
                            </div>

                            {expandedSources[message.id] ? (
                              <div className="space-y-2">
                                {message.sourceItems.map((source) => (
                                  <div
                                    className="rounded-2xl border border-slate-100 bg-white px-3 py-3 shadow-sm"
                                    key={`detail-${source.chunk_id || `${source.document_id}-${source.page_number}`}`}
                                  >
                                    <div className="flex flex-wrap items-center gap-2">
                                      <span className="text-sm font-medium text-slate-900">{source.filename}</span>
                                      {source.page_number ? (
                                        <span className="rounded-full border border-slate-200 bg-slate-100 px-2 py-0.5 text-[11px] text-slate-800">
                                          Page {source.page_number}
                                        </span>
                                      ) : null}
                                    </div>
                                    {source.source_path ? (
                                      <p className="mt-1 truncate text-xs text-slate-400">{source.source_path}</p>
                                    ) : null}
                                    <p className="mt-2 text-sm leading-6 text-slate-600">{source.excerpt}</p>
                                  </div>
                                ))}
                              </div>
                            ) : null}
                          </div>
                        ) : null}
                      </div>

                    </div>
                  )
                })}

                <div ref={messagesEndRef} />
              </div>
            )}
          </section>

          <footer className="bg-gradient-to-t from-[#f7f9fb] via-[#f7f9fb] to-transparent px-4 pb-6 pt-2 md:px-6">
            <div className="mx-auto w-full max-w-[940px]">
              <form onSubmit={handleSubmit}>
                <div className="group flex h-16 items-center gap-3 rounded-full border border-slate-200 bg-white px-5 shadow-sm transition focus-within:border-slate-500 focus-within:ring-2 focus-within:ring-slate-300/60">
                  <div className="shrink-0 text-slate-400">
                    <Search size={18} />
                  </div>
                  <textarea
                    className="h-8 min-h-0 flex-1 resize-none overflow-y-auto border-none bg-transparent pt-[8px] text-[17px] leading-6 text-slate-800 outline-none placeholder:text-slate-400"
                    disabled={isLoading}
                    id="chat-input"
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleComposerKeyDown}
                    placeholder="Ask anything..."
                    ref={textareaRef}
                    rows={1}
                    value={input}
                  />
                  <button
                    className="shrink-0 rounded-full p-2 text-slate-400 transition hover:bg-slate-50 hover:text-slate-600"
                    onClick={() => setUploadPanelOpen(true)}
                    type="button"
                  >
                    <Upload size={17} />
                  </button>
                  <button
                    className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-slate-900 text-white shadow-sm transition hover:bg-black disabled:cursor-not-allowed disabled:opacity-60"
                    disabled={isLoading || !input.trim()}
                    type="submit"
                  >
                    {isLoading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                  </button>
                </div>
              </form>
            </div>
          </footer>
        </main>
      </div>

      <div
        className={`fixed right-4 top-4 z-[60] w-[min(420px,calc(100vw-2rem))] rounded-[24px] border border-slate-200 bg-white shadow-[0_20px_60px_rgba(15,23,42,0.12)] transition-all duration-300 ${
          uploadPanelOpen ? 'translate-y-0 opacity-100' : 'pointer-events-none -translate-y-4 opacity-0'
        }`}
      >
        <div className="border-b border-slate-100 px-5 py-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Uploads</p>
              <h3 className="mt-1 text-lg font-semibold text-slate-900">PDF Manager</h3>
              <p className="mt-1 text-sm text-slate-500">Upload files without taking space away from your chats.</p>
            </div>
            <button
              aria-label="Close uploads panel"
              className="rounded-full p-2 text-slate-400 transition hover:bg-slate-50 hover:text-slate-600"
              onClick={() => setUploadPanelOpen(false)}
              type="button"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        <div className="max-h-[calc(100vh-8rem)] overflow-y-auto px-5 py-4">
          <div className="rounded-2xl border border-slate-100 bg-slate-50 px-4 py-4">
            <div className="flex items-center gap-2 text-sm font-medium text-slate-700">
              {isUploading ? <Loader2 className="animate-spin" size={16} /> : <Upload size={16} />}
              {isUploading ? 'Uploading PDF...' : 'Choose a PDF from your files'}
            </div>
            <input
              accept=".pdf"
              className="mt-3 block w-full cursor-pointer rounded-xl border border-slate-200 bg-white px-3 py-3 text-sm text-slate-700 file:mr-4 file:rounded-lg file:border-0 file:bg-slate-200 file:px-3 file:py-2 file:text-sm file:font-medium file:text-slate-800 hover:border-slate-300"
              disabled={isUploading}
              onChange={handleFileUpload}
              ref={fileInputRef}
              type="file"
            />
          </div>

          {uploadStatus ? (
            <div
              className={`mt-4 rounded-2xl border px-3 py-3 text-sm ${
                uploadStatus.kind === 'success'
                  ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                  : uploadStatus.kind === 'error'
                    ? 'border-red-200 bg-red-50 text-red-700'
                    : 'border-slate-200 bg-slate-100 text-slate-700'
              }`}
            >
              <div className="flex items-start gap-2">
                {uploadStatus.kind === 'progress' ? (
                  <Loader2 className="mt-0.5 animate-spin" size={16} />
                ) : uploadStatus.kind === 'success' ? (
                  <Check className="mt-0.5" size={16} />
                ) : (
                  <Clock3 className="mt-0.5" size={16} />
                )}
                <p>{uploadStatus.message}</p>
              </div>
            </div>
          ) : null}

          <div className="mt-5">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">Documents</p>
                <p className="mt-1 text-xs text-slate-500">These are the PDFs currently available for answers.</p>
              </div>
              <button
                className="text-xs text-slate-400 transition hover:text-slate-600"
                onClick={() => void loadIndexedDocuments()}
                type="button"
              >
                Refresh
              </button>
            </div>
            <div className="mt-2 space-y-2">
              {isLoadingDocuments ? (
                <div className="rounded-2xl border border-dashed border-slate-200 px-3 py-3 text-xs text-slate-500">
                  Loading indexed documents...
                </div>
              ) : indexedDocuments.length === 0 ? (
                <div className="rounded-2xl border border-dashed border-slate-200 px-3 py-3 text-xs text-slate-500">
                  No documents uploaded yet.
                </div>
              ) : (
                indexedDocuments.map((document) => (
                  <div className="rounded-2xl border border-slate-100 bg-slate-50 px-3 py-3" key={document.document_id}>
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium text-slate-800">{document.filename}</p>
                        <p className="mt-1 text-xs text-slate-500">
                          {document.chunk_count} chunks{document.page_count ? ` • ${document.page_count} pages` : ''}
                        </p>
                      </div>
                      <button
                        aria-label={`Delete ${document.filename}`}
                        className="rounded-xl border border-red-100 bg-red-50 p-2 text-red-500 transition hover:border-red-200 hover:bg-red-100 disabled:cursor-not-allowed disabled:opacity-60"
                        disabled={deletingDocumentId === document.document_id}
                        onClick={() => void handleDeleteDocument(document)}
                        type="button"
                      >
                        {deletingDocumentId === document.document_id ? (
                          <Loader2 className="animate-spin" size={14} />
                        ) : (
                          <Trash2 size={14} />
                        )}
                      </button>
                    </div>
                    <p className="mt-2 truncate text-xs text-slate-400">{document.source_path}</p>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
