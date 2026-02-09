"use client";

import { useEffect, useState } from "react";
import { fetchReports } from "@/lib/api";
import ReportCard from "@/components/ReportCard";
import type { Report } from "@/lib/api";

export default function ReportsPage() {
    const [reports, setReports] = useState<Report[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const loadReports = async () => {
            try {
                const data = await fetchReports();
                setReports(data);
            } catch (err) {
                setError("Failed to load reports");
                console.error(err);
            } finally {
                setLoading(false);
            }
        };

        loadReports();
    }, []);

    const sortedReports = [...reports].sort(
        (a, b) =>
            new Date(b.timestamp || new Date().toISOString()).getTime() -
            new Date(a.timestamp || new Date().toISOString()).getTime()
    );

    const highRiskCount = reports.filter((r) => (r.risk_score || 0) >= 7).length;
    const mediumRiskCount = reports.filter(
        (r) => (r.risk_score || 0) >= 4 && (r.risk_score || 0) < 7
    ).length;
    const lowRiskCount = reports.filter((r) => (r.risk_score || 0) < 4).length;

    return (
        <div className="min-h-screen bg-gray-950 text-gray-100">
            <div className="max-w-6xl mx-auto px-4 py-12">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-4xl font-bold mb-2">Call Reports</h1>
                    <p className="text-gray-400">
                        Review and analyze financial service call recordings
                    </p>
                </div>

                {/* Stats */}
                {reports.length > 0 && (
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
                        <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
                            <p className="text-sm text-gray-400">Total Calls</p>
                            <p className="text-2xl font-bold">{reports.length}</p>
                        </div>
                        <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
                            <p className="text-sm text-gray-400">Avg Risk Score</p>
                            <p className="text-2xl font-bold">
                                {(
                                    reports.reduce((sum, r) => sum + (r.risk_score || 0), 0) /
                                    reports.length
                                ).toFixed(1)}
                            </p>
                        </div>
                        <div className="bg-red-900/20 border border-red-700 rounded-lg p-4">
                            <p className="text-sm text-red-200">High Risk</p>
                            <p className="text-2xl font-bold text-red-100">{highRiskCount}</p>
                        </div>
                        <div className="bg-yellow-900/20 border border-yellow-700 rounded-lg p-4">
                            <p className="text-sm text-yellow-200">Medium Risk</p>
                            <p className="text-2xl font-bold text-yellow-100">
                                {mediumRiskCount}
                            </p>
                        </div>
                    </div>
                )}

                {/* Loading state */}
                {loading && (
                    <div className="text-center py-12">
                        <p className="text-gray-400">Loading reports...</p>
                    </div>
                )}

                {/* Error state */}
                {error && (
                    <div className="bg-red-900/20 border border-red-700 rounded-lg p-4 mb-6">
                        <p className="text-red-200">{error}</p>
                    </div>
                )}

                {/* Empty state */}
                {!loading && reports.length === 0 && (
                    <div className="text-center py-12 bg-gray-900 rounded-lg border border-gray-700">
                        <p className="text-gray-400">No reports yet. Upload an audio file to get started.</p>
                    </div>
                )}

                {/* Reports grid */}
                {!loading && reports.length > 0 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {sortedReports.map((report, idx) => (
                            <ReportCard
                                key={report.id || `report-${idx}`}
                                report={report}
                            />
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
