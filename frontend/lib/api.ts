const API_BASE = "http://127.0.0.1:8000";

export interface Report {
    id: string;
    filename: string;
    timestamp: string;
    risk_score: number;
    transcript?: string;
    violations?: string[];
    obligations?: string[];
    stress_timeline?: { time: string; stress: number }[];
    agent_segments?: { speaker: string; time: string; text: string; risk_keywords?: string[] }[];
    customer_segments?: { speaker: string; time: string; text: string; risk_keywords?: string[] }[];
}

/**
 * Fetch all reports from backend
 */
export async function fetchReports(): Promise<Report[]> {
    try {
        const res = await fetch(`${API_BASE}/reports`);
        if (!res.ok) throw new Error(`Failed to fetch reports: ${res.status}`);
        const data = await res.json();

        // Handle if response is directly an array
        if (Array.isArray(data)) {
            return data.map(mapBackendReport);
        }

        // Handle if response is an object with reports property
        if (data && typeof data === "object" && Array.isArray(data.reports)) {
            return data.reports.map(mapBackendReport);
        }

        // Handle if response has data property
        if (data && typeof data === "object" && Array.isArray(data.data)) {
            return data.data.map(mapBackendReport);
        }

        return [];
    } catch (error) {
        console.error("Error fetching reports:", error);
        return [];
    }
}

/**
 * Map backend report structure to frontend Report interface
 */
function mapBackendReport(backendReport: any): Report {
    const overallRisk = backendReport.overall_risk || {};
    // Backend score is 0-100, convert to 0-10 scale for frontend display
    const backendScore = typeof overallRisk.score === "number" ? overallRisk.score : (backendReport.risk_score || 0);
    const riskScore = Math.min(backendScore / 10, 10);  // Convert 0-100 to 0-10

    const violations = extractViolations(backendReport);
    const obligations = extractObligations(backendReport);
    const agentSegments = extractSegments(backendReport, "agent");
    const customerSegments = extractSegments(backendReport, "customer");
    const stressTimeline = extractStressTimeline(backendReport);

    // Debug logging
    console.log("Report mapping debug:", {
        filename: backendReport.filename,
        obligationsCount: obligations.length,
        obligations: obligations.slice(0, 3),
        agentSegmentsCount: agentSegments.length,
        customerSegmentsCount: customerSegments.length,
        violationsCount: violations.length,
    });

    return {
        id: backendReport.filename || backendReport.id || `report-${Date.now()}`,
        filename: backendReport.audio_file || backendReport.filename || "Unknown Call",
        timestamp: backendReport.processed_at || backendReport.timestamp || new Date().toISOString(),
        risk_score: riskScore,
        transcript: backendReport.transcript,
        violations,
        obligations,
        stress_timeline: stressTimeline,
        agent_segments: agentSegments,
        customer_segments: customerSegments,
    };
}

/**
 * Extract violations from backend report
 */
function extractViolations(report: any): string[] {
    const violations: Set<string> = new Set();

    // From profanity findings
    if (report.profanity_findings && Array.isArray(report.profanity_findings)) {
        report.profanity_findings.forEach((p: any) => {
            if (p.phrase) {
                violations.add(`Prohibited phrase: "${p.phrase}"`);
            }
        });
    }

    // From regulatory compliance
    if (report.regulatory_compliance?.violations && Array.isArray(report.regulatory_compliance.violations)) {
        report.regulatory_compliance.violations.forEach((v: any) => {
            if (v.description) {
                violations.add(v.description);
            }
        });
    }

    // From risk factors
    if (report.overall_risk?.risk_factors && Array.isArray(report.overall_risk.risk_factors)) {
        report.overall_risk.risk_factors.forEach((factor: string) => {
            if (factor) violations.add(factor);
        });
    }

    return Array.from(violations);
}

/**
 * Extract obligations from backend report
 */
function extractObligations(report: any): string[] {
    const obligations: Set<string> = new Set();

    // From obligation analysis - obligations array with quote field
    if (report.obligation_analysis && typeof report.obligation_analysis === "object") {
        const obligationsArray = report.obligation_analysis.obligations;
        if (obligationsArray && Array.isArray(obligationsArray)) {
            obligationsArray.forEach((o: any) => {
                if (o && typeof o === "object") {
                    // Try different field names for the obligation text
                    const quote = o.quote || o.commitment || o.description || o.title || o.text || "";
                    if (quote && typeof quote === "string" && quote.trim()) {
                        obligations.add(quote);
                    }
                } else if (typeof o === "string" && o.trim()) {
                    obligations.add(o);
                }
            });
        }
    }

    // From obligation sentences - array of sentence objects or strings
    if (report.obligation_sentences && Array.isArray(report.obligation_sentences)) {
        report.obligation_sentences.forEach((sent: any) => {
            if (typeof sent === "string" && sent.trim()) {
                obligations.add(sent);
            } else if (sent && typeof sent === "object") {
                // Handle if it has a sentence field
                const text = sent.sentence || sent.text || "";
                if (text && typeof text === "string" && text.trim()) {
                    obligations.add(text);
                }
            }
        });
    }

    // From intent classification (might contain obligations)
    if (report.intent_classification && typeof report.intent_classification === "object") {
        const intents = report.intent_classification;
        if (intents.commitments && Array.isArray(intents.commitments)) {
            intents.commitments.forEach((c: any) => {
                const text = typeof c === "string" ? c : (c.text || c.description || c.quote || "");
                if (text && typeof text === "string" && text.trim()) {
                    obligations.add(text);
                }
            });
        }
    }

    return Array.from(obligations).filter((o) => o && o.trim().length > 0);
}

/**
 * Extract stress timeline from emotion analysis
 */
function extractStressTimeline(report: any): { time: string; stress: number }[] {
    const timeline: { time: string; stress: number }[] = [];

    if (report.emotion_analysis?.stress_timeline && Array.isArray(report.emotion_analysis.stress_timeline)) {
        return report.emotion_analysis.stress_timeline
            .map((item: any) => ({
                time: item.timestamp || item.time || "0:00",
                stress: typeof item.stress_level === "number" ? Math.min(item.stress_level, 10) : 0,
            }))
            .filter((item: { time: string; stress: number }) => item.stress >= 0);
    }

    // Fallback: create dummy timeline if emotion data exists
    if (report.emotion_analysis?.overall_stress) {
        const stressLevelMap = { low: 3, medium: 5, high: 8 };
        const baseStress =
            stressLevelMap[report.emotion_analysis.overall_stress as keyof typeof stressLevelMap] || 5;

        return [
            { time: "0:00", stress: Math.round(baseStress * 0.7) },
            { time: "0:30", stress: Math.round(baseStress * 0.85) },
            { time: "1:00", stress: baseStress },
            { time: "1:30", stress: Math.round(baseStress * 0.9) },
            { time: "2:00", stress: Math.round(baseStress * 0.8) },
        ];
    }

    return timeline;
}

/**
 * Extract transcript segments by speaker
 */
function extractSegments(report: any, speaker: "agent" | "customer"): { speaker: string; time: string; text: string; risk_keywords?: string[] }[] {
    const allSegments: (any & { originalOrder: number })[] = [];
    const speakersMap = new Map<string, string>(); // Map original speaker name to agent/customer

    if (!report.segments || !Array.isArray(report.segments)) {
        return [];
    }

    // First pass: identify unique speakers and tag all segments with original order
    const uniqueSpeakers = new Set<string>();
    report.segments.forEach((seg: any, index: number) => {
        allSegments.push({ ...seg, originalOrder: index });
        if (seg.speaker && typeof seg.speaker === "string") {
            uniqueSpeakers.add(seg.speaker);
        }
    });

    // Map speakers to agent/customer roles
    let agentCount = 0;
    let customerCount = 0;
    uniqueSpeakers.forEach((spk) => {
        if (spk.toLowerCase().includes("agent") || spk.toLowerCase().includes("speaker 1")) {
            speakersMap.set(spk, "Agent");
            agentCount++;
        } else if (spk.toLowerCase().includes("customer") || spk.toLowerCase().includes("speaker 2")) {
            speakersMap.set(spk, "Customer");
            customerCount++;
        } else if (agentCount === 0) {
            speakersMap.set(spk, "Agent");
            agentCount++;
        } else {
            speakersMap.set(spk, "Customer");
            customerCount++;
        }
    });

    // Second pass: extract and format segments
    const result: { speaker: string; time: string; text: string; risk_keywords?: string[] }[] = [];

    allSegments.forEach((seg, index) => {
        const mappedSpeaker = speakersMap.get(seg.speaker) || "Agent";
        const isRequestedSpeaker = mappedSpeaker.toLowerCase().includes(speaker.toLowerCase());

        if (isRequestedSpeaker && seg.text) {
            // Format timestamp - prefer actual timestamp, fallback to calculated time
            let timeStr = "0:00";
            let timeValue = 0;

            if (typeof seg.start === "number" && seg.start >= 0) {
                const minutes = Math.floor(seg.start / 60);
                const seconds = Math.floor(seg.start % 60);
                timeStr = `${minutes}:${seconds.toString().padStart(2, "0")}`;
                timeValue = seg.start;
            } else if (typeof seg.end === "number" && seg.end >= 0) {
                const minutes = Math.floor(seg.end / 60);
                const seconds = Math.floor(seg.end % 60);
                timeStr = `${minutes}:${seconds.toString().padStart(2, "0")}`;
                timeValue = seg.end;
            } else if (seg.timestamp) {
                timeStr = seg.timestamp;
            } else {
                // Calculate time based on index in original order - interleave properly
                const timeInSeconds = index * 3;  // Assume ~3 seconds per segment
                const minutes = Math.floor(timeInSeconds / 60);
                const seconds = Math.floor(timeInSeconds % 60);
                timeStr = `${minutes}:${seconds.toString().padStart(2, "0")}`;
                timeValue = timeInSeconds;
            }

            result.push({
                speaker: mappedSpeaker,
                time: timeStr,
                text: seg.text || "",
                risk_keywords: seg.risk_keywords || seg.keywords || [],
            });
        }
    });

    return result;
}

/**
 * Fetch a single report by ID
 */
export async function fetchReport(id: string): Promise<Report | null> {
    try {
        const res = await fetch(`${API_BASE}/reports/${id}`);
        if (!res.ok) throw new Error(`Failed to fetch report: ${res.status}`);
        const data = await res.json();
        return mapBackendReport(data);
    } catch (error) {
        console.error("Error fetching report:", error);
        return null;
    }
}

/**
 * Upload and analyze audio file through the full pipeline (L1→L2→L3).
 * Reads an SSE (Server-Sent Events) stream from the backend so the UI
 * can display progress while each layer runs.
 */
export async function analyzeAudio(
    file: File,
    language?: string,
    onProgress?: (message: string, stage: string) => void,
): Promise<Report | null> {
    try {
        const formData = new FormData();
        formData.append("file", file);

        const params = new URLSearchParams();
        if (language && language !== "auto") {
            params.set("language", language);
        }
        const qs = params.toString();
        const url = `${API_BASE}/analyze${qs ? `?${qs}` : ""}`;

        const res = await fetch(url, {
            method: "POST",
            body: formData,
        });

        if (!res.ok) {
            const errBody = await res.text();
            throw new Error(`Analysis failed (${res.status}): ${errBody}`);
        }

        // Read the SSE stream
        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let report: Report | null = null;
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split("\n\n");
            buffer = events.pop() || ""; // last element may be incomplete

            for (const event of events) {
                const dataLine = event
                    .split("\n")
                    .find((line: string) => line.startsWith("data: "));
                if (!dataLine) continue;

                try {
                    const data = JSON.parse(dataLine.slice(6));

                    if (data.type === "progress" && onProgress) {
                        onProgress(data.message, data.stage);
                    } else if (data.type === "complete") {
                        report = mapBackendReport(data.report);
                    } else if (data.type === "error") {
                        throw new Error(data.message);
                    }
                } catch (parseErr: any) {
                    // If it's our own thrown error, re-throw
                    if (parseErr.message && !parseErr.message.includes("JSON")) {
                        throw parseErr;
                    }
                    console.warn("SSE parse warning:", parseErr);
                }
            }
        }

        return report;
    } catch (error: any) {
        console.error("Error analyzing audio:", error);
        throw error;
    }
}

/**
 * Fetch caller history (Backboard memory feature)
 */
export async function fetchCallerHistory(callerId: string): Promise<Report[]> {
    try {
        const res = await fetch(`${API_BASE}/backboard/history/${callerId}`);
        if (!res.ok) throw new Error(`Failed to fetch caller history: ${res.status}`);
        const data = await res.json();

        // Handle if response is directly an array
        if (Array.isArray(data)) {
            return data;
        }

        // Handle if response is an object with reports/data property
        if (data && typeof data === "object" && Array.isArray(data.reports)) {
            return data.reports;
        }

        if (data && typeof data === "object" && Array.isArray(data.data)) {
            return data.data;
        }

        return [];
    } catch (error) {
        console.error("Error fetching caller history:", error);
        return [];
    }
}