"use client";

interface Violation {
    id?: string;
    type: string;
    severity: "high" | "medium" | "low";
    description: string;
    timestamp?: string;
    speaker?: string;
    quote?: string;
}

interface ViolationsListProps {
    violations: Violation[] | string[];
}

export default function ViolationsList({ violations }: ViolationsListProps) {
    if (!violations || violations.length === 0) {
        return (
            <div className="bg-gray-900 rounded-lg border border-gray-700 p-6 text-center">
                <p className="text-gray-400">✓ No violations detected</p>
            </div>
        );
    }

    // Normalize violations to objects
    const normalizedViolations: Violation[] = violations.map((v) => {
        if (typeof v === "string") {
            return {
                type: "Policy Violation",
                severity: "high",
                description: v,
            };
        }
        return v;
    });

    const severityColors: Record<string, string> = {
        high: "border-l-4 border-red-500 bg-red-900/20",
        medium: "border-l-4 border-yellow-500 bg-yellow-900/20",
        low: "border-l-4 border-blue-500 bg-blue-900/20",
    };

    const severityBadge: Record<string, string> = {
        high: "bg-red-900 text-red-100",
        medium: "bg-yellow-900 text-yellow-100",
        low: "bg-blue-900 text-blue-100",
    };

    return (
        <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
            <div className="p-4 border-b border-gray-700 bg-gray-800">
                <h3 className="text-lg font-semibold text-gray-100">
                    Violations ({normalizedViolations.length})
                </h3>
            </div>
            <div className="divide-y divide-gray-700 max-h-100 overflow-y-auto">
                {normalizedViolations.map((violation, idx) => (
                    <div
                        key={idx}
                        className={`p-4 ${severityColors[violation.severity] || severityColors.medium}`}
                    >
                        <div className="flex justify-between items-start mb-2">
                            <h4 className="font-semibold text-gray-100">{violation.type}</h4>
                            <span
                                className={`px-2 py-1 rounded text-xs font-medium ${severityBadge[violation.severity]
                                    }`}
                            >
                                {violation.severity.toUpperCase()}
                            </span>
                        </div>
                        <p className="text-gray-300 text-sm mb-2">{violation.description}</p>
                        {violation.timestamp && (
                            <p className="text-xs text-gray-400">
                                Time: {violation.timestamp}
                            </p>
                        )}
                        {violation.speaker && (
                            <p className="text-xs text-gray-400">
                                Speaker: {violation.speaker}
                            </p>
                        )}
                        {violation.quote && (
                            <p className="text-xs text-gray-400 italic mt-2">
                                "{violation.quote}"
                            </p>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
