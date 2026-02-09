"use client";

import { Report } from "@/lib/api";
import Link from "next/link";

interface ReportCardProps {
    report: Report;
}

function getRiskBadgeColor(score: number): string {
    if (score >= 7) return "bg-red-900 text-red-100";
    if (score >= 4) return "bg-yellow-900 text-yellow-100";
    return "bg-green-900 text-green-100";
}

function getRiskBadgeLabel(score: number): string {
    if (score >= 7) return "HIGH";
    if (score >= 4) return "MEDIUM";
    return "LOW";
}

export default function ReportCard({ report }: ReportCardProps) {
    const riskScore = report.risk_score || 0;
    const reportId = report.id || "unknown";
    const timestamp = new Date(report.timestamp || new Date().toISOString()).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
    });

    return (
        <Link href={`/reports/${reportId}`}>
            <div className="p-4 rounded-lg border border-gray-700 bg-gray-900 hover:bg-gray-800 transition-colors cursor-pointer">
                <div className="flex justify-between items-start mb-3">
                    <div>
                        <h3 className="font-semibold text-gray-100 truncate">
                            {report.filename || "Unknown Call"}
                        </h3>
                        <p className="text-sm text-gray-400">{timestamp}</p>
                    </div>
                    <span
                        className={`px-3 py-1 rounded-full text-sm font-medium whitespace-nowrap ml-2 ${getRiskBadgeColor(
                            riskScore
                        )}`}
                    >
                        {getRiskBadgeLabel(riskScore)} ({riskScore.toFixed(1)})
                    </span>
                </div>

                {report.violations && report.violations.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-700">
                        <p className="text-xs text-gray-400 mb-1">
                            {report.violations.length} violation
                            {report.violations.length !== 1 ? "s" : ""}
                        </p>
                    </div>
                )}
            </div>
        </Link>
    );
}
