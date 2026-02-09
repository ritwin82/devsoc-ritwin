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

// ── Component ─────────────────────────────────────────────────────────────

export default function RealtimePage() {
  // State
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
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

  // Refs
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);
  const alertsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll transcript
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcriptChunks]);

  // Auto-scroll alerts
  useEffect(() => {
    alertsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [alerts]);

  // Timer
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

    try {
      // 1. Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      });
      streamRef.current = stream;

      // 2. Connect WebSocket
      const ws = new WebSocket("ws://127.0.0.1:8000/ws/realtime");
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        // Send start message
        ws.send(JSON.stringify({ type: "start", language }));
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleServerMessage(msg);
      };

      ws.onerror = (e) => {
        console.error("WebSocket error:", e);
        setError("WebSocket connection failed. Is the backend running?");
      };

      ws.onclose = () => {
        setIsConnected(false);
      };

      // 3. Wait for WebSocket to be ready before starting MediaRecorder
      ws.addEventListener("message", function onReady(event) {
        const msg = JSON.parse(event.data);
        if (msg.type === "ready") {
          ws.removeEventListener("message", onReady);

          // 4. Start MediaRecorder with 5-second chunks
          const recorder = new MediaRecorder(stream, {
            mimeType: "audio/webm;codecs=opus",
          });
          mediaRecorderRef.current = recorder;

          recorder.ondataavailable = async (e) => {
            if (e.data.size > 0 && ws.readyState === WebSocket.OPEN) {
              // Convert blob to base64
              const reader = new FileReader();
              reader.onloadend = () => {
                const base64 = (reader.result as string).split(",")[1];
                ws.send(JSON.stringify({ type: "audio", data: base64 }));
              };
              reader.readAsDataURL(e.data);
            }
          };

          recorder.start(5000); // 5-second chunks
          setIsRecording(true);
        }
      });
    } catch (err: any) {
      setError(err.message || "Failed to start recording");
      console.error("Start recording error:", err);
    }
  }, [language]);

  // ── Stop Recording ──────────────────────────────────────────────────

  const stopRecording = useCallback(() => {
    // Stop MediaRecorder
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }

    // Stop mic stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
    }

    // Send stop to backend
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "stop" }));
    }

    setIsRecording(false);
  }, []);

  // ── Handle Server Messages ──────────────────────────────────────────

  const handleServerMessage = (msg: any) => {
    switch (msg.type) {
      case "chunk_result":
        setChunkCount(msg.total_chunks || 0);
        if (msg.transcript) {
          setTranscriptChunks((prev) => [
            ...prev,
            {
              timestamp: msg.timestamp,
              text: msg.transcript,
              chunk_id: msg.chunk_id,
            },
          ]);
        }
        if (msg.full_transcript) {
          setFullTranscript(msg.full_transcript);
        }
        if (msg.alerts && msg.alerts.length > 0) {
          setAlerts((prev) => [...prev, ...msg.alerts]);
        }
        break;

      case "compliance_update":
        if (msg.data && typeof msg.data === "object") {
          setCompliance(msg.data);
        }
        break;

      case "session_ended":
        setSessionSummary(msg.summary);
        break;

      case "batch_processing_complete":
        setBatchReport(msg);
        break;

      case "batch_processing_failed":
        setError(`Batch processing failed: ${msg.error}`);
        break;

      case "error":
        setError(msg.message);
        break;
    }
  };

  // ── Alert Severity Colors ───────────────────────────────────────────

  const getAlertColor = (severity: string, type: string) => {
    if (type === "financial_entity") return "border-blue-700 bg-blue-900/20 text-blue-200";
    if (severity === "high") return "border-red-700 bg-red-900/20 text-red-200";
    if (severity === "medium") return "border-yellow-700 bg-yellow-900/20 text-yellow-200";
    if (severity === "info") return "border-gray-700 bg-gray-800 text-gray-300";
    return "border-gray-700 bg-gray-800 text-gray-300";
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case "pii": return "🔒";
      case "prohibited": return "🚫";
      case "profanity": return "⚠️";
      case "obligation": return "📋";
      case "financial_entity": return "💰";
      default: return "ℹ️";
    }
  };

  // ── Risk Score Color ────────────────────────────────────────────────

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
            ← Back to Home
          </Link>
          <h1 className="text-4xl font-bold mb-2">Real-Time Call Analysis</h1>
          <p className="text-gray-400">
            Record a call and get live transcription with compliance monitoring
          </p>
        </div>

        {/* Controls */}
        <div className="bg-gray-900 rounded-lg border border-gray-700 p-6 mb-8">
          <div className="flex flex-wrap items-center gap-4">
            {/* Language Selector */}
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

            {/* Record Button */}
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

            {/* Status */}
            <div className="text-right">
              <p className="text-sm text-gray-400">
                {isRecording ? "Recording..." : isConnected ? "Connected" : "Ready"}
              </p>
              <p className="text-2xl font-mono font-bold">
                {formatTime(elapsed)}
              </p>
              <p className="text-xs text-gray-500">{chunkCount} chunks processed</p>
            </div>
          </div>

          {/* Connection Status */}
          <div className="mt-3 flex items-center gap-2 text-sm">
            <span className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-500" : isRecording ? "bg-yellow-500 animate-pulse" : "bg-gray-500"}`} />
            <span className="text-gray-400">
              {isConnected ? "Backend connected" : isRecording ? "Connecting..." : "Disconnected"}
            </span>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 mb-6">
            <p className="text-red-200">{error}</p>
          </div>
        )}

        {/* Main Grid: Transcript | Alerts */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Live Transcript (2 cols) */}
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
            <div className="p-4 max-h-[500px] overflow-y-auto space-y-3">
              {transcriptChunks.length === 0 && (
                <p className="text-gray-500 text-center py-8">
                  {isRecording
                    ? "Listening for speech..."
                    : "Press Start Recording to begin"}
                </p>
              )}
              {transcriptChunks.map((chunk, idx) => (
                <div key={idx} className="flex gap-3">
                  <span className="text-xs text-gray-500 font-mono whitespace-nowrap pt-1">
                    {formatTime(Math.round(chunk.timestamp))}
                  </span>
                  <p className="text-gray-200 text-sm leading-relaxed">
                    {chunk.text}
                  </p>
                </div>
              ))}
              <div ref={transcriptEndRef} />
            </div>
          </div>

          {/* Alerts Sidebar (1 col) */}
          <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
            <div className="p-4 border-b border-gray-700 bg-gray-800 flex justify-between items-center">
              <h3 className="text-lg font-semibold">Alerts</h3>
              {alerts.length > 0 && (
                <span className="bg-red-900 text-red-100 px-2 py-0.5 rounded-full text-xs font-bold">
                  {alerts.filter((a) => a.severity === "high").length} critical
                </span>
              )}
            </div>
            <div className="p-3 max-h-[500px] overflow-y-auto space-y-2">
              {alerts.length === 0 && (
                <p className="text-gray-500 text-center py-8 text-sm">
                  No alerts yet
                </p>
              )}
              {alerts
                .filter((a) => a.type !== "financial_entity") // hide info-level by default
                .map((alert, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded border text-sm ${getAlertColor(alert.severity, alert.type)}`}
                >
                  <div className="flex items-start gap-2">
                    <span>{getAlertIcon(alert.type)}</span>
                    <div className="flex-1">
                      <p className="font-medium">{alert.message}</p>
                      <p className="text-xs opacity-70 mt-1">
                        {formatTime(Math.round(alert.timestamp))}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
              <div ref={alertsEndRef} />
            </div>
          </div>
        </div>

        {/* Compliance Status */}
        {compliance && (
          <div className="bg-gray-900 rounded-lg border border-gray-700 p-6 mb-8">
            <h3 className="text-lg font-semibold mb-4">
              Regulatory Compliance Check
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-400 mb-1">Compliance Score</p>
                <p className={`text-4xl font-bold ${getRiskColor(compliance.compliance_score || 0)}`}>
                  {compliance.compliance_score ?? "—"}
                </p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-400 mb-1">Status</p>
                <p className="text-xl font-semibold capitalize">
                  {compliance.overall_compliance || "—"}
                </p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-400 mb-1">Violations</p>
                <p className="text-xl font-semibold text-red-400">
                  {compliance.violations?.length ?? 0}
                </p>
              </div>
            </div>
            {compliance.violations && compliance.violations.length > 0 && (
              <div className="mt-4 space-y-2">
                {compliance.violations.map((v, idx) => (
                  <div key={idx} className="border-l-4 border-red-500 bg-red-900/20 p-3 rounded">
                    <p className="text-sm font-medium text-red-200">{v.regulation}</p>
                    <p className="text-xs text-gray-400 italic mt-1">"{v.quote}"</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Session Summary (after stop) */}
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

        {/* Batch Report Link */}
        {batchReport && (
          <div className="bg-green-900/20 border border-green-700 rounded-lg p-6 mb-8">
            <h3 className="text-lg font-semibold text-green-200 mb-2">
              Full Analysis Complete
            </h3>
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
                View Full Report →
              </Link>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
