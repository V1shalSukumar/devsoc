"use client";

interface TranscriptSegment {
    speaker: string;
    time: string;
    text: string;
    risk_keywords?: string[];
}

interface TranscriptViewerProps {
    agentSegments?: TranscriptSegment[];
    customerSegments?: TranscriptSegment[];
    prohibitedPhrases?: string[];
    obligations?: string[];
}

/**
 * Escape special regex characters in a string
 */
function escapeRegex(str: string): string {
    return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export default function TranscriptViewer({
    agentSegments = [],
    customerSegments = [],
    prohibitedPhrases = [],
    obligations = [],
}: TranscriptViewerProps) {
    // Helper function to convert "M:SS" to total seconds for comparison
    const timeToSeconds = (timeStr: string): number => {
        if (!timeStr) return 0;
        const parts = timeStr.split(":");
        if (parts.length === 2) {
            const minutes = parseInt(parts[0]) || 0;
            const seconds = parseInt(parts[1]) || 0;
            return minutes * 60 + seconds;
        }
        return 0;
    };

    // Combine and interleave segments while preserving order
    const allSegments: (TranscriptSegment & { originalIndex: number; timeInSeconds: number })[] = [];

    // Add agent segments
    if (agentSegments && Array.isArray(agentSegments)) {
        agentSegments.forEach((s, idx) => {
            allSegments.push({
                ...s,
                speaker: "Agent",
                originalIndex: idx,
                timeInSeconds: timeToSeconds(s.time),
            });
        });
    }

    // Add customer segments
    if (customerSegments && Array.isArray(customerSegments)) {
        customerSegments.forEach((s, idx) => {
            allSegments.push({
                ...s,
                speaker: "Customer",
                originalIndex: 1000 + idx,  // Offset to keep customer segments separate
                timeInSeconds: timeToSeconds(s.time),
            });
        });
    }

    // Sort by time, then by original order to interleave speakers
    allSegments.sort((a, b) => {
        if (a.timeInSeconds !== b.timeInSeconds) {
            return a.timeInSeconds - b.timeInSeconds;
        }
        return a.originalIndex - b.originalIndex;
    });

    const highlightText = (text: string, keywords: string[] = []) => {
        if (!text || text.length === 0) return text;

        let highlightedText = text;

        try {
            // Highlight prohibited phrases in RED
            if (prohibitedPhrases && prohibitedPhrases.length > 0) {
                prohibitedPhrases.forEach((phrase) => {
                    if (phrase && phrase.length > 0 && phrase.length < 200) {
                        try {
                            const escapedPhrase = escapeRegex(phrase);
                            const regex = new RegExp(`(${escapedPhrase})`, "gi");
                            highlightedText = highlightedText.replace(
                                regex,
                                '<span class="bg-red-900 text-red-100 px-1 rounded">$1</span>'
                            );
                        } catch (e) {
                            // Skip this phrase if regex fails
                            console.warn("Failed to highlight phrase:", phrase);
                        }
                    }
                });
            }

            // Highlight obligations in YELLOW
            if (obligations && obligations.length > 0) {
                obligations.forEach((obligation) => {
                    if (obligation && obligation.length > 0 && obligation.length < 200) {
                        try {
                            const escapedObligation = escapeRegex(obligation);
                            const regex = new RegExp(`(${escapedObligation})`, "gi");
                            highlightedText = highlightedText.replace(
                                regex,
                                '<span class="bg-yellow-900 text-yellow-100 px-1 rounded">$1</span>'
                            );
                        } catch (e) {
                            // Skip this obligation if regex fails
                            console.warn("Failed to highlight obligation:", obligation);
                        }
                    }
                });
            }
        } catch (e) {
            console.warn("Error highlighting text:", e);
        }

        return highlightedText;
    };

    if (allSegments.length === 0) {
        return (
            <div className="bg-gray-900 rounded-lg border border-gray-700 p-6 text-center">
                <p className="text-gray-400">No transcript available</p>
            </div>
        );
    }

    return (
        <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
            <div className="p-4 border-b border-gray-700 bg-gray-800">
                <h3 className="text-lg font-semibold text-gray-100">Transcript</h3>
            </div>
            <div className="divide-y divide-gray-700 max-h-[500px] overflow-y-auto">
                {allSegments.map((segment, idx) => (
                    <div key={idx} className="p-4 hover:bg-gray-800 transition-colors">
                        <div className="flex justify-between items-baseline mb-2">
                            <span
                                className={`font-semibold text-sm ${segment.speaker === "Agent"
                                        ? "text-blue-400"
                                        : "text-purple-400"
                                    }`}
                            >
                                {segment.speaker}
                            </span>
                            <span className="text-xs text-gray-500">{segment.time}</span>
                        </div>
                        <p
                            className="text-gray-300 leading-relaxed text-sm"
                            dangerouslySetInnerHTML={{
                                __html: highlightText(segment.text, segment.risk_keywords),
                            }}
                        />
                    </div>
                ))}
            </div>
        </div>
    );
}
