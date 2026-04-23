'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Upload, FileText, Loader2, Bot, User, Trash2 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: number
}

interface Thread {
  id: string
  title: string
  createdAt: Date
}

const createThreadId = () => `thread-${Date.now()}`

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isLoadingThread, setIsLoadingThread] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [threads, setThreads] = useState<Thread[]>([])
  const [currentThreadId, setCurrentThreadId] = useState<string>('')
  const [uploadStatus, setUploadStatus] = useState<string | null>(null)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001'

  useEffect(() => {
    // Generate thread ID on mount
    const threadId = createThreadId()
    setCurrentThreadId(threadId)

    // Load threads from localStorage
    const savedThreads = localStorage.getItem('docbot-threads')
    if (savedThreads) {
      setThreads(JSON.parse(savedThreads))
    }
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const saveThread = (threadId: string, firstMessage: string) => {
    const newThread: Thread = {
      id: threadId,
      title: firstMessage.slice(0, 50) + (firstMessage.length > 50 ? '...' : ''),
      createdAt: new Date()
    }
    const updatedThreads = [newThread, ...threads.filter(t => t.id !== threadId)]
    setThreads(updatedThreads)
    localStorage.setItem('docbot-threads', JSON.stringify(updatedThreads))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading || !currentThreadId) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // Save thread on first message
    if (messages.length === 0) {
      saveThread(currentThreadId, userMessage.content)
    }

    // Create placeholder for assistant message
    const assistantMessageId = (Date.now() + 1).toString()
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      sources: 0
    }
    setMessages(prev => [...prev, assistantMessage])

    try {
      const response = await fetch(`${API_URL}/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream'
        },
        body: JSON.stringify({
          question: userMessage.content,
          thread_id: currentThreadId
        })
      })

      if (!response.ok) throw new Error('Failed to get response')
      if (!response.body) throw new Error('Streaming response body is missing')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let fullContent = ''
      let sources = 0
      let buffer = ''

      const applyAssistantUpdate = (content: string, nextSources: number) => {
        setMessages(prev => prev.map(msg =>
          msg.id === assistantMessageId
            ? { ...msg, content, sources: nextSources }
            : msg
        ))
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

        if (eventType === 'metadata') {
          sources = parsed.sources || 0
          applyAssistantUpdate(fullContent, sources)
          return false
        }

        if (eventType === 'token') {
          fullContent += parsed.token || ''
          applyAssistantUpdate(fullContent, sources)
          return false
        }

        if (eventType === 'error') {
          throw new Error(parsed.message || 'Streaming request failed')
        }

        if (eventType === 'done') {
          applyAssistantUpdate(fullContent, sources)
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

          if (!eventBlock.trim()) {
            continue
          }

          const isDone = processEvent(eventBlock)
          if (isDone) {
            buffer = ''
            break
          }
        }
      }

      const finalChunk = decoder.decode()
      if (finalChunk) {
        buffer += finalChunk
      }

      if (buffer.trim()) {
        processEvent(buffer.trim())
      }
    } catch (error) {
      // Update the assistant message with error
      setMessages(prev => prev.map(msg =>
        msg.id === assistantMessageId
          ? { ...msg, content: error instanceof Error ? error.message : 'Sorry, I encountered an error. Please make sure the backend is running.' }
          : msg
      ))
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !file.name.endsWith('.pdf')) {
      setUploadStatus('Please select a PDF file')
      return
    }

    setIsUploading(true)
    setUploadStatus(`Uploading ${file.name}...`)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch(`${API_URL}/ingest/upload`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) throw new Error('Upload failed')

      const data = await response.json()
      setUploadStatus(`${file.name} uploaded! ${data.chunk_count} chunks created.`)

      setTimeout(() => setUploadStatus(null), 5000)
    } catch (error) {
      setUploadStatus('Upload failed. Is the backend running?')
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const startNewChat = () => {
    const newThreadId = createThreadId()
    setCurrentThreadId(newThreadId)
    setMessages([])
  }

  const loadThread = async (threadId: string) => {
    setCurrentThreadId(threadId)
    setIsLoadingThread(true)
    setMessages([]) // Clear first

    try {
      console.log('Loading thread:', threadId)

      const response = await fetch(`${API_URL}/threads/${threadId}/history`)
      console.log('Response status:', response.status)

      if (response.ok) {
        const data = await response.json()
        console.log('Thread data:', data)

        if (data.messages && data.messages.length > 0) {
          const loadedMessages: Message[] = data.messages.map((msg: any, index: number) => ({
            id: `${threadId}-${index}`,
            role: msg.role,
            content: msg.content
          }))
          setMessages(loadedMessages)
        }
      }
    } catch (error) {
      console.error('Failed to load thread history:', error)
    } finally {
      setIsLoadingThread(false)
    }
  }

  const deleteThread = (threadId: string) => {
    const updatedThreads = threads.filter(t => t.id !== threadId)
    setThreads(updatedThreads)
    localStorage.setItem('docbot-threads', JSON.stringify(updatedThreads))
    if (currentThreadId === threadId) {
      startNewChat()
    }
  }

  return (
    <div className="flex h-screen bg-[#0a0a0a]">
      {/* Sidebar */}
      <div className="w-64 bg-[#111] border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <button
            onClick={startNewChat}
            className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            <FileText size={18} />
            New Chat
          </button>
        </div>

        {/* Thread List */}
        <div className="flex-1 overflow-y-auto p-2">
          <p className="text-xs text-gray-500 px-2 py-1">Recent Chats</p>
          {threads.map(thread => (
            <div
              key={thread.id}
              className={`group flex items-center gap-2 p-2 rounded-lg cursor-pointer hover:bg-gray-800 ${
                currentThreadId === thread.id ? 'bg-gray-800' : ''
              }`}
              onClick={() => loadThread(thread.id)}
            >
              <span className="flex-1 text-sm text-gray-300 truncate">
                {thread.title}
              </span>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  deleteThread(thread.id)
                }}
                className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-700 rounded"
              >
                <Trash2 size={14} className="text-gray-400" />
              </button>
            </div>
          ))}
        </div>

        {/* Upload Section */}
        <div className="p-4 border-t border-gray-800">
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            className="hidden"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="w-full py-2 px-4 border border-gray-700 hover:bg-gray-800 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
          >
            {isUploading ? (
              <Loader2 size={18} className="animate-spin" />
            ) : (
              <Upload size={18} />
            )}
            Upload PDF
          </button>
          {uploadStatus && (
            <p className="text-xs text-gray-400 mt-2 text-center">{uploadStatus}</p>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="h-14 border-b border-gray-800 flex items-center px-6">
          <Bot className="text-blue-500 mr-2" size={24} />
          <h1 className="text-lg font-semibold">DocBot</h1>
          <span className="ml-2 text-sm text-gray-500">AI Document Assistant</span>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoadingThread ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-500">
              <Loader2 size={48} className="mb-4 text-blue-500 animate-spin" />
              <p className="text-gray-400">Loading conversation...</p>
            </div>
          ) : messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-500">
              <Bot size={48} className="mb-4 text-blue-500" />
              <h2 className="text-xl font-medium text-gray-300 mb-2">Welcome to DocBot</h2>
              <p className="text-center max-w-md">
                Upload a PDF document and ask questions about it.
                I'll help you find answers from your documents.
              </p>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map(message => (
                <div
                  key={message.id}
                  className={`flex gap-4 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
                >
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    message.role === 'user' ? 'bg-blue-600' : 'bg-gray-700'
                  }`}>
                    {message.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                  </div>
                  <div className={`flex-1 ${message.role === 'user' ? 'text-right' : ''}`}>
                    <div className={`inline-block p-4 rounded-2xl max-w-full ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-800 text-gray-100'
                    }`}>
                      {message.role === 'assistant' ? (
                        message.content ? (
                          <div className="prose prose-invert max-w-none">
                            <ReactMarkdown
                              remarkPlugins={[remarkMath]}
                              rehypePlugins={[rehypeKatex]}
                            >
                              {message.content}
                            </ReactMarkdown>
                          </div>
                        ) : (
                          <Loader2 className="animate-spin text-blue-500" size={20} />
                        )
                      ) : (
                        message.content
                      )}
                    </div>
                    {message.sources !== undefined && message.sources > 0 && (
                      <p className="text-xs text-gray-500 mt-1">
                        Sources: {message.sources} document chunks
                      </p>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && messages.length > 0 && messages[messages.length - 1].role === 'user' && (
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center">
                    <Bot size={18} />
                  </div>
                  <div className="bg-gray-800 p-4 rounded-2xl">
                    <Loader2 className="animate-spin text-blue-500" size={20} />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-gray-800 p-4">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            <div className="flex gap-4">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question about your documents..."
                className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 transition-colors"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 rounded-xl transition-colors"
              >
                <Send size={20} />
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  )
}
