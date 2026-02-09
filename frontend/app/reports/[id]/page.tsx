"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { fetchReport, fetchCallerHistory } from "@/lib/api";
import StressChart from "@/components/StressChart";
import TranscriptViewer from "@/components/TranscriptViewer";
import ViolationsList from "@/components/ViolationsList";
import ObligationsList from "@/components/ObligationsList";
import type { Report } from "@/lib/api";

export default function ReportDetailPage() {
    const params = useParams();
    const reportId = params.id as string;

    const [report, setReport] = useState<Report | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [showCallerHistory, setShowCallerHistory] = useState(false);
    const [callerHistory, setCallerHistory] = useState<Report[]>([]);
    const [loadingHistory, setLoadingHistory] = useState(false);

    useEffect(() => {
        const loadReport = async () => {
            try {
                const data = await fetchReport(reportId);
                if (!data) throw new Error("Report not found");
                setReport(data);
            } catch (err) {
                setError("Failed to load report");
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        loadReport();
    }, [reportId]);

    const handleShowCallerHistory = async () => {
        if (!report) return;

        setShowCallerHistory(true);
        setLoadingHistory(true);

        try {
            // Extract caller ID from report (you may need to adjust this based on your actual data structure)
            const callerId = report.id || "unknown";
            const history = await fetchCallerHistory(callerId);
            setCallerHistory(history);
        } catch (err) {
            console.error("Failed to fetch caller history:", err);
        } finally {
            setLoadingHistory(false);
        }
    };

    function getRiskBadgeColor(score: number): string {
        if (score >= 7) return "bg-red-600 text-red-100";
        if (score >= 4) return "bg-yellow-600 text-yellow-100";
        return "bg-green-600 text-green-100";
    }

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-950 flex items-center justify-center">
                <p className="text-gray-400">Loading report...</p>
            </div>
        );
    }

    if (error || !report) {
        return (
            <div className="min-h-screen bg-gray-950 text-gray-100">
                <div className="max-w-6xl mx-auto px-4 py-12">
                    <div className="bg-red-900/20 border border-red-700 rounded-lg p-6">
                        <p className="text-red-200">{error || "Report not found"}</p>
                    </div>
                    <Link
                        href="/reports"
                        className="mt-6 inline-block px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                    >
                        ← Back to Reports
                    </Link>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-950 text-gray-100">
            <div className="max-w-6xl mx-auto px-4 py-8">
                {/* Header */}
                <div className="mb-8">
                    <Link
                        href="/reports"
                        className="text-blue-400 hover:text-blue-300 mb-4 inline-block"
                    >
                        ← Back to Reports
                    </Link>

                    <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6">
                        <div>
                            <h1 className="text-4xl font-bold mb-2">{report.filename || "Unknown Call"}</h1>
                            <p className="text-gray-400">
                                {new Date(report.timestamp || new Date().toISOString()).toLocaleDateString("en-US", {
                                    month: "long",
                                    day: "numeric",
                                    year: "numeric",
                                    hour: "2-digit",
                                    minute: "2-digit",
                                })}
                            </p>
                        </div>
                        <span
                            className={`px-6 py-3 rounded-lg text-xl font-bold whitespace-nowrap ${getRiskBadgeColor(
                                report.risk_score || 0
                            )}`}
                        >
                            Risk Score: {(report.risk_score || 0).toFixed(1)}
                        </span>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex flex-wrap gap-3">
                        <button
                            onClick={handleShowCallerHistory}
                            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg transition-colors text-sm"
                        >
                            📞 View Caller History
                        </button>
                    </div>
                </div>

                {/* Caller History Modal */}
                {showCallerHistory && (
                    <div className="mb-8 bg-purple-900/20 border border-purple-700 rounded-lg p-6">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-xl font-semibold">Caller History</h3>
                            <button
                                onClick={() => setShowCallerHistory(false)}
                                className="text-gray-400 hover:text-gray-200"
                            >
                                ✕
                            </button>
                        </div>

                        {loadingHistory && <p className="text-gray-400">Loading...</p>}

                        {!loadingHistory && callerHistory.length === 0 && (
                            <p className="text-gray-400">No previous calls found</p>
                        )}

                        {!loadingHistory && callerHistory.length > 0 && (
                            <div className="space-y-3 max-h-64 overflow-y-auto">
                                {callerHistory.map((call, idx) => (
                                    <Link
                                        key={call.id || `call-${idx}`}
                                        href={`/reports/${call.id || '#'}`}
                                        className="block p-3 bg-gray-800 rounded border border-gray-700 hover:border-purple-500 transition-colors"
                                    >
                                        <p className="font-medium text-gray-100">{call.filename || "Unknown Call"}</p>
                                        <p className="text-sm text-gray-400">
                                            {new Date(call.timestamp || new Date().toISOString()).toLocaleDateString()}
                                        </p>
                                    </Link>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {/* Main Content Grid */}
                <div className="space-y-8">
                    {/* Stress Timeline */}
                    {report.stress_timeline && report.stress_timeline.length > 0 && (
                        <section>
                            <StressChart data={report.stress_timeline} />
                        </section>
                    )}

                    {/* Transcript */}
                    <section>
                        <TranscriptViewer
                            agentSegments={report.agent_segments}
                            customerSegments={report.customer_segments}
                            prohibitedPhrases={report.violations}
                            obligations={report.obligations}
                        />
                    </section>

                    {/* Grid Layout for Violations and Obligations */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        {/* Violations */}
                        <section>
                            <ViolationsList violations={report.violations || []} />
                        </section>

                        {/* Obligations */}
                        <section>
                            <ObligationsList obligations={report.obligations || []} />
                        </section>
                    </div>
                </div>
            </div>
        </div>
    );
}
