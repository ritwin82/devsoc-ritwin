"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
    sendQuery, fetchQuerySuggestions,
    type QueryResponse, type QuerySuggestion
} from "@/lib/api";

interface ChatMessage {
    id: string;
    role: "user" | "bot";
    content: string;
    timestamp: Date;
    dataContext?: { total_calls: number; data_source: string };
}

export default function QueryPage() {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [suggestions, setSuggestions] = useState<QuerySuggestion[]>([]);
    const [sessionId] = useState(() => `query_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    // Load suggestions on mount
    useEffect(() => {
        fetchQuerySuggestions().then(setSuggestions);
    }, []);

    // Auto-scroll to bottom
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    // Focus input on load
    useEffect(() => {
        inputRef.current?.focus();
    }, []);

    const handleSend = async (question?: string) => {
        const q = (question || input).trim();
        if (!q || loading) return;

        const userMsg: ChatMessage = {
            id: `user_${Date.now()}`,
            role: "user",
            content: q,
            timestamp: new Date(),
        };

        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setLoading(true);

        try {
            const response = await sendQuery(q, sessionId);

            const botMsg: ChatMessage = {
                id: `bot_${Date.now()}`,
                role: "bot",
                content: response?.answer || "I wasn't able to process that question. Please try again.",
                timestamp: new Date(),
                dataContext: response?.data_context,
            };

            setMessages((prev) => [...prev, botMsg]);
        } catch {
            const errorMsg: ChatMessage = {
                id: `err_${Date.now()}`,
                role: "bot",
                content: "⚠️ Something went wrong. Please make sure the backend is running and try again.",
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, errorMsg]);
        } finally {
            setLoading(false);
            inputRef.current?.focus();
        }
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const categoryColors: Record<string, string> = {
        overview: "border-blue-600 text-blue-300 hover:bg-blue-900/40",
        compliance: "border-green-600 text-green-300 hover:bg-green-900/40",
        risk: "border-red-600 text-red-300 hover:bg-red-900/40",
        sentiment: "border-purple-600 text-purple-300 hover:bg-purple-900/40",
        privacy: "border-yellow-600 text-yellow-300 hover:bg-yellow-900/40",
        intent: "border-cyan-600 text-cyan-300 hover:bg-cyan-900/40",
        emotion: "border-pink-600 text-pink-300 hover:bg-pink-900/40",
        outcome: "border-orange-600 text-orange-300 hover:bg-orange-900/40",
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex flex-col">
            {/* Header */}
            <div className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm">
                <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-xl shadow-lg shadow-violet-500/20">
                            🤖
                        </div>
                        <div>
                            <h1 className="text-lg font-bold text-white">Analytics Query Bot</h1>
                            <p className="text-xs text-slate-400">Ask anything about your call analysis data</p>
                        </div>
                    </div>
                    <Link
                        href="/"
                        className="px-3 py-1.5 text-sm bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg transition border border-slate-700"
                    >
                        ← Back
                    </Link>
                </div>
            </div>

            {/* Chat Area */}
            <div className="flex-1 overflow-y-auto">
                <div className="max-w-4xl mx-auto px-4 py-6 space-y-4">
                    {/* Welcome message */}
                    {messages.length === 0 && (
                        <div className="text-center py-16">
                            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-4xl mx-auto mb-6 shadow-xl shadow-violet-500/20">
                                🤖
                            </div>
                            <h2 className="text-2xl font-bold text-white mb-2">
                                Hello! I&apos;m your Analytics Assistant
                            </h2>
                            <p className="text-slate-400 mb-8 max-w-lg mx-auto">
                                I can answer questions about your analyzed calls — compliance scores,
                                risk levels, sentiments, PII detection, and more. Try asking one of these:
                            </p>

                            {/* Suggestion chips */}
                            <div className="flex flex-wrap justify-center gap-2 max-w-2xl mx-auto">
                                {suggestions.map((s, i) => (
                                    <button
                                        key={i}
                                        onClick={() => handleSend(s.text)}
                                        className={`px-3 py-2 rounded-lg text-sm border transition-all duration-200 cursor-pointer ${categoryColors[s.category] || "border-slate-600 text-slate-300 hover:bg-slate-800"
                                            }`}
                                    >
                                        {s.text}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Messages */}
                    {messages.map((msg) => (
                        <div
                            key={msg.id}
                            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                        >
                            <div
                                className={`max-w-[80%] rounded-2xl px-4 py-3 ${msg.role === "user"
                                    ? "bg-gradient-to-r from-violet-600 to-indigo-600 text-white rounded-br-md"
                                    : "bg-slate-800/80 border border-slate-700 text-slate-200 rounded-bl-md"
                                    }`}
                            >
                                {msg.role === "bot" && (
                                    <div className="flex items-center gap-2 mb-2 text-xs text-slate-400">
                                        <span className="text-violet-400">🤖 Analytics Bot</span>
                                        {msg.dataContext && (
                                            <span className="bg-slate-700/60 px-2 py-0.5 rounded-full">
                                                {msg.dataContext.total_calls} calls analyzed
                                            </span>
                                        )}
                                    </div>
                                )}
                                <div className="whitespace-pre-wrap text-sm leading-relaxed">
                                    {msg.content}
                                </div>
                                <div className={`text-xs mt-2 ${msg.role === "user" ? "text-white/50" : "text-slate-500"
                                    }`}>
                                    {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                                </div>
                            </div>
                        </div>
                    ))}

                    {/* Typing indicator */}
                    {loading && (
                        <div className="flex justify-start">
                            <div className="bg-slate-800/80 border border-slate-700 rounded-2xl rounded-bl-md px-4 py-3">
                                <div className="flex items-center gap-2 mb-1 text-xs text-slate-400">
                                    <span className="text-violet-400">🤖 Analytics Bot</span>
                                </div>
                                <div className="flex items-center gap-1.5">
                                    <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                                    <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: "150ms" }} />
                                    <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce" style={{ animationDelay: "300ms" }} />
                                </div>
                            </div>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* Suggestion chips (after first message) */}
            {messages.length > 0 && messages.length < 4 && (
                <div className="border-t border-slate-800/50">
                    <div className="max-w-4xl mx-auto px-4 py-2">
                        <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
                            {suggestions.slice(0, 5).map((s, i) => (
                                <button
                                    key={i}
                                    onClick={() => handleSend(s.text)}
                                    disabled={loading}
                                    className={`px-3 py-1.5 rounded-lg text-xs border whitespace-nowrap flex-shrink-0 transition-all duration-200 cursor-pointer disabled:opacity-50 ${categoryColors[s.category] || "border-slate-600 text-slate-300 hover:bg-slate-800"
                                        }`}
                                >
                                    {s.text}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* Input Area */}
            <div className="border-t border-slate-800 bg-slate-900/90 backdrop-blur-sm">
                <div className="max-w-4xl mx-auto px-4 py-4">
                    <div className="flex gap-3 items-center">
                        <input
                            ref={inputRef}
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask about your call data..."
                            disabled={loading}
                            className="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:border-violet-500/50 transition disabled:opacity-50"
                        />
                        <button
                            onClick={() => handleSend()}
                            disabled={loading || !input.trim()}
                            className="px-5 py-3 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white rounded-xl font-medium transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-violet-500/20 hover:shadow-violet-500/30 cursor-pointer"
                        >
                            {loading ? (
                                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                </svg>
                            ) : (
                                "Send"
                            )}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
