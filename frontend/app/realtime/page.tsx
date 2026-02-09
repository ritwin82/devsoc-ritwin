"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import Link from "next/link";

// ── Types ─────────────────────────────────────────────────────────────────

interface Alert {
  type: string;
  severity: string;
  message: string;
  timestamp: number;
  keywords?: string[];
}

interface TranscriptChunk {
  timestamp: number;
  text: string;
  chunk_id: number;
}

interface ComplianceData {
  compliance_score?: number;
  overall_compliance?: string;
  violations?: { regulation: string; quote: string; severity: string }[];
  recommendations?: string[];
}

// ── Helpers ───────────────────────────────────────────────────────────────

function getMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "audio/webm";
  if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus"))
    return "audio/webm;codecs=opus";
  if (MediaRecorder.isTypeSupported("audio/webm")) return "audio/webm";
  if (MediaRecorder.isTypeSupported("audio/mp4")) return "audio/mp4";
  return "audio/webm";
}

// ── Component ─────────────────────────────────────────────────────────────

export default function RealtimePage() {
  // State
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [wsReady, setWsReady] = useState(false);
  const [language, setLanguage] = useState("en");
  const [transcriptChunks, setTranscriptChunks] = useState<TranscriptChunk[]>([]);
  const [fullTranscript, setFullTranscript] = useState("");
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [compliance, setCompliance] = useState<ComplianceData | null>(null);
  const [sessionSummary, setSessionSummary] = useState<any>(null);
  const [batchReport, setBatchReport] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [chunkCount, setChunkCount] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [status, setStatus] = useState("Ready");

  // Refs (for use inside callbacks that may close over stale state)
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const isRecordingRef = useRef(false);
  const wsReadyRef = useRef(false);
  const chunkTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const alertsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcriptChunks]);
  useEffect(() => {
    alertsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [alerts]);

  // Elapsed timer
  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRecording]);

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  // ── Send one audio blob to backend via WS ───────────────────────────

  const sendAudioChunk = useCallback((blob: Blob) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN || !wsReadyRef.current) {
      return;
    }
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64 = (reader.result as string).split(",")[1];
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "audio", data: base64 }));
      }
    };
    reader.readAsDataURL(blob);
  }, []);

  // ── Record one chunk using stop/restart for valid standalone files ──

  const recordChunk = useCallback(() => {
    if (!isRecordingRef.current || !streamRef.current) return;

    const stream = streamRef.current;
    const mimeType = getMimeType();

    try {
      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;

      const chunks: Blob[] = [];
      recorder.ondataavailable = (e: BlobEvent) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = () => {
        // Each stop produces a complete standalone file
        if (chunks.length > 0) {
          const blob = new Blob(chunks, { type: mimeType });
          sendAudioChunk(blob);
        }
        // Schedule next chunk if still recording
        if (isRecordingRef.current) {
          recordChunk();
        }
      };

      recorder.start();

      // Stop after 5 seconds → triggers onstop → sends complete file
      chunkTimerRef.current = setTimeout(() => {
        if (recorder.state === "recording") {
          recorder.stop();
        }
      }, 5000);
    } catch (err) {
      console.error("MediaRecorder error:", err);
    }
  }, [sendAudioChunk]);

  // ── Handle messages from backend ────────────────────────────────────

  const handleServerMessage = useCallback((msg: any) => {
    switch (msg.type) {
      case "ready":
        wsReadyRef.current = true;
        setWsReady(true);
        setStatus("Listening...");
        break;

      case "chunk_result":
        setChunkCount(msg.total_chunks || 0);
        if (msg.transcript) {
          setTranscriptChunks((prev) => [
            ...prev,
            { timestamp: msg.timestamp, text: msg.transcript, chunk_id: msg.chunk_id },
          ]);
          setStatus("Transcribing...");
        }
        if (msg.full_transcript) setFullTranscript(msg.full_transcript);
        if (msg.alerts && msg.alerts.length > 0) {
          setAlerts((prev) => [...prev, ...msg.alerts]);
        }
        break;

      case "compliance_update":
        if (msg.data && typeof msg.data === "object") setCompliance(msg.data);
        break;

      case "session_ended":
        setSessionSummary(msg.summary);
        setStatus("Session ended — running batch analysis...");
        break;

      case "batch_processing_started":
        setStatus("Running full pipeline on recorded audio...");
        break;

      case "batch_processing_complete":
        setBatchReport(msg);
        setStatus("Complete");
        break;

      case "batch_processing_failed":
        setError(`Batch processing failed: ${msg.error}`);
        setStatus("Batch failed");
        break;

      case "error":
        setError(msg.message);
        break;
    }
  }, []);

  // ── Start Recording ─────────────────────────────────────────────────

  const startRecording = useCallback(async () => {
    setError(null);
    setTranscriptChunks([]);
    setFullTranscript("");
    setAlerts([]);
    setCompliance(null);
    setSessionSummary(null);
    setBatchReport(null);
    setChunkCount(0);
    setElapsed(0);
    setStatus("Requesting microphone...");

    try {
      // 1. Microphone
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, sampleRate: 16000 },
      });
      streamRef.current = stream;
      setStatus("Connecting to backend...");

      // 2. WebSocket
      const ws = new WebSocket("ws://127.0.0.1:8000/ws/realtime");
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        ws.send(JSON.stringify({ type: "start", language }));
        setStatus("Initializing...");
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleServerMessage(msg);

        // Begin chunk recording once backend signals ready
        if (msg.type === "ready") {
          isRecordingRef.current = true;
          setIsRecording(true);
          setStatus("Recording...");
          recordChunk();
        }
      };

      ws.onerror = () => {
        setError("WebSocket connection failed. Is the backend running on port 8000?");
        setStatus("Connection failed");
      };

      ws.onclose = () => {
        setIsConnected(false);
        setWsReady(false);
        wsReadyRef.current = false;
      };
    } catch (err: any) {
      setError(err.message || "Failed to start recording");
      setStatus("Error");
    }
  }, [language, handleServerMessage, recordChunk]);

  // ── Stop Recording ──────────────────────────────────────────────────

  const stopRecording = useCallback(() => {
    isRecordingRef.current = false;

    if (chunkTimerRef.current) {
      clearTimeout(chunkTimerRef.current);
      chunkTimerRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop(); // triggers onstop → sends last chunk
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    // Delay "stop" so the last chunk has time to send
    setTimeout(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "stop" }));
      }
    }, 800);

    setIsRecording(false);
    setStatus("Stopping...");
  }, []);

  // ── UI helpers ──────────────────────────────────────────────────────

  const getAlertColor = (severity: string, type: string) => {
    if (type === "fraud" || type === "social_engineering") return "border-orange-700 bg-orange-900/20 text-orange-200";
    if (type === "financial_entity") return "border-blue-700 bg-blue-900/20 text-blue-200";
    if (severity === "high") return "border-red-700 bg-red-900/20 text-red-200";
    if (severity === "medium") return "border-yellow-700 bg-yellow-900/20 text-yellow-200";
    return "border-gray-700 bg-gray-800 text-gray-300";
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case "pii": return "🔒";
      case "prohibited": return "🚫";
      case "profanity": return "⚠️";
      case "obligation": return "📋";
      case "financial_entity": return "💰";
      case "fraud": return "🚨";
      case "social_engineering": return "🎭";
      default: return "ℹ️";
    }
  };

  const getRiskColor = (score: number) => {
    if (score >= 80) return "text-green-400";
    if (score >= 50) return "text-yellow-400";
    return "text-red-400";
  };

  // ── Render ──────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link href="/" className="text-blue-400 hover:text-blue-300 mb-4 inline-block">
            &larr; Back to Home
          </Link>
          <h1 className="text-4xl font-bold mb-2">Real-Time Call Analysis</h1>
          <p className="text-gray-400">
            Record a call and get live transcription with compliance monitoring
          </p>
        </div>

        {/* Controls */}
        <div className="bg-gray-900 rounded-lg border border-gray-700 p-6 mb-8">
          <div className="flex flex-wrap items-center gap-4">
            <div>
              <label className="text-sm text-gray-400 block mb-1">Language</label>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                disabled={isRecording}
                className="bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm"
              >
                <option value="en">English</option>
                <option value="hi">Hindi</option>
                <option value="ru">Russian</option>
                <option value="auto">Auto-detect</option>
              </select>
            </div>

            <div className="flex-1 flex justify-center">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  className="px-8 py-3 bg-red-600 hover:bg-red-700 rounded-full text-lg font-semibold transition-colors flex items-center gap-2"
                >
                  <span className="w-3 h-3 bg-red-300 rounded-full" />
                  Start Recording
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="px-8 py-3 bg-gray-600 hover:bg-gray-700 rounded-full text-lg font-semibold transition-colors flex items-center gap-2 animate-pulse"
                >
                  <span className="w-3 h-3 bg-white rounded-sm" />
                  Stop Recording
                </button>
              )}
            </div>

            <div className="text-right min-w-36">
              <p className="text-sm text-gray-400">{status}</p>
              <p className="text-2xl font-mono font-bold">{formatTime(elapsed)}</p>
              <p className="text-xs text-gray-500">{chunkCount} chunks</p>
            </div>
          </div>

          <div className="mt-3 flex items-center gap-2 text-sm">
            <span className={`w-2 h-2 rounded-full ${wsReady ? "bg-green-500" : isConnected ? "bg-yellow-500 animate-pulse" : "bg-gray-500"}`} />
            <span className="text-gray-400">
              {wsReady ? "Backend connected" : isConnected ? "Initializing..." : "Disconnected"}
            </span>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 mb-6">
            <p className="text-red-200">{error}</p>
          </div>
        )}

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Live Transcript */}
          <div className="lg:col-span-2 bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
            <div className="p-4 border-b border-gray-700 bg-gray-800 flex justify-between items-center">
              <h3 className="text-lg font-semibold">Live Transcript</h3>
              {isRecording && (
                <span className="flex items-center gap-1 text-red-400 text-sm">
                  <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                  LIVE
                </span>
              )}
            </div>
            <div className="p-4 max-h-125 overflow-y-auto space-y-3">
              {transcriptChunks.length === 0 && (
                <p className="text-gray-500 text-center py-8">
                  {isRecording ? "Listening for speech..." : "Press Start Recording to begin"}
                </p>
              )}
              {transcriptChunks.map((chunk, idx) => (
                <div key={idx} className="flex gap-3">
                  <span className="text-xs text-gray-500 font-mono whitespace-nowrap pt-1">
                    {formatTime(Math.round(chunk.timestamp))}
                  </span>
                  <p className="text-gray-200 text-sm leading-relaxed">{chunk.text}</p>
                </div>
              ))}
              <div ref={transcriptEndRef} />
            </div>
          </div>

          {/* Alerts Sidebar */}
          <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
            <div className="p-4 border-b border-gray-700 bg-gray-800 flex justify-between items-center">
              <h3 className="text-lg font-semibold">Alerts</h3>
              {alerts.filter((a) => a.severity === "high" || a.type === "fraud" || a.type === "social_engineering").length > 0 && (
                <span className="bg-red-900 text-red-100 px-2 py-0.5 rounded-full text-xs font-bold">
                  {alerts.filter((a) => a.severity === "high" || a.type === "fraud" || a.type === "social_engineering").length} critical
                </span>
              )}
            </div>
            <div className="p-3 max-h-125 overflow-y-auto space-y-2">
              {alerts.length === 0 && (
                <p className="text-gray-500 text-center py-8 text-sm">No alerts yet</p>
              )}
              {alerts
                .filter((a) => a.type !== "financial_entity")
                .map((alert, idx) => (
                  <div
                    key={idx}
                    className={`p-3 rounded border text-sm ${getAlertColor(alert.severity, alert.type)}`}
                  >
                    <div className="flex items-start gap-2">
                      <span>{getAlertIcon(alert.type)}</span>
                      <div className="flex-1">
                        <p className="font-medium">{alert.message}</p>
                        <p className="text-xs opacity-70 mt-1">{formatTime(Math.round(alert.timestamp))}</p>
                      </div>
                    </div>
                  </div>
                ))}
              <div ref={alertsEndRef} />
            </div>
          </div>
        </div>

        {/* Compliance */}
        {compliance && (
          <div className="bg-gray-900 rounded-lg border border-gray-700 p-6 mb-8">
            <h3 className="text-lg font-semibold mb-4">Regulatory Compliance Check</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-400 mb-1">Compliance Score</p>
                <p className={`text-4xl font-bold ${getRiskColor(compliance.compliance_score || 0)}`}>
                  {compliance.compliance_score ?? "—"}
                </p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-400 mb-1">Status</p>
                <p className="text-xl font-semibold capitalize">{compliance.overall_compliance || "—"}</p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-400 mb-1">Violations</p>
                <p className="text-xl font-semibold text-red-400">{compliance.violations?.length ?? 0}</p>
              </div>
            </div>
            {compliance.violations && compliance.violations.length > 0 && (
              <div className="mt-4 space-y-2">
                {compliance.violations.map((v, idx) => (
                  <div key={idx} className="border-l-4 border-red-500 bg-red-900/20 p-3 rounded">
                    <p className="text-sm font-medium text-red-200">{v.regulation}</p>
                    <p className="text-xs text-gray-400 italic mt-1">&quot;{v.quote}&quot;</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Session Summary */}
        {sessionSummary && (
          <div className="bg-gray-900 rounded-lg border border-gray-700 p-6 mb-8">
            <h3 className="text-lg font-semibold mb-4">Session Summary</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-800 rounded-lg p-4">
                <p className="text-sm text-gray-400">Duration</p>
                <p className="text-2xl font-bold">{formatTime(Math.round(sessionSummary.duration_seconds))}</p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <p className="text-sm text-gray-400">Chunks</p>
                <p className="text-2xl font-bold">{sessionSummary.total_chunks}</p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <p className="text-sm text-gray-400">Total Alerts</p>
                <p className="text-2xl font-bold">{sessionSummary.total_alerts}</p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <p className="text-sm text-gray-400">PII Detected</p>
                <p className="text-2xl font-bold text-red-400">{sessionSummary.alert_breakdown?.pii || 0}</p>
              </div>
            </div>
          </div>
        )}

        {/* Batch Report */}
        {batchReport && (
          <div className="bg-green-900/20 border border-green-700 rounded-lg p-6 mb-8">
            <h3 className="text-lg font-semibold text-green-200 mb-2">Full Analysis Complete</h3>
            <p className="text-sm text-green-300 mb-3">
              The recorded call has been processed through all layers.
            </p>
            <div className="flex items-center gap-4">
              <span className={`text-2xl font-bold ${getRiskColor(batchReport.overall_risk?.score || 0)}`}>
                Risk Score: {batchReport.overall_risk?.score ?? "—"}
              </span>
              <Link
                href="/reports"
                className="px-4 py-2 bg-green-700 hover:bg-green-600 rounded-lg text-sm transition-colors"
              >
                View Full Report &rarr;
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
