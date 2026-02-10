"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
    PieChart, Pie, Cell, ResponsiveContainer,
    Area, AreaChart
} from "recharts";
import {
    fetchAnalyticsSummary, fetchAnalyticsTrends,
    AnalyticsSummary, AnalyticsTrends
} from "@/lib/api";

// Color palettes
const COLORS = {
    compliance: ["#ef4444", "#f97316", "#eab308", "#22c55e"],
    risk: { high: "#ef4444", medium: "#f97316", low: "#22c55e", unknown: "#6b7280" },
    intent: ["#3b82f6", "#8b5cf6", "#ec4899", "#14b8a6", "#f59e0b", "#6366f1"],
    chart: { primary: "#3b82f6", secondary: "#8b5cf6", accent: "#22c55e" }
};

export default function AnalyticsPage() {
    const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
    const [trends, setTrends] = useState<AnalyticsTrends | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function loadData() {
            try {
                const [summaryData, trendsData] = await Promise.all([
                    fetchAnalyticsSummary(),
                    fetchAnalyticsTrends()
                ]);
                setSummary(summaryData);
                setTrends(trendsData);
            } catch (err) {
                setError("Failed to load analytics data");
                console.error(err);
            } finally {
                setLoading(false);
            }
        }
        loadData();
    }, []);

    if (loading) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
                <div className="text-white text-xl animate-pulse">Loading analytics...</div>
            </div>
        );
    }

    if (error || !summary) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
                <div className="text-center">
                    <div className="text-red-400 text-xl mb-4">{error || "No data available"}</div>
                    <Link href="/" className="text-blue-400 hover:underline">← Back to Home</Link>
                </div>
            </div>
        );
    }

    // Transform data for charts
    const riskData = Object.entries(summary.risk_level_distribution || {}).map(([name, value]) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value,
        color: COLORS.risk[name as keyof typeof COLORS.risk] || "#6b7280"
    }));

    const intentData = Object.entries(summary.intent_distribution || {}).map(([name, value], i) => ({
        name: name.charAt(0).toUpperCase() + name.slice(1),
        value,
        color: COLORS.intent[i % COLORS.intent.length]
    }));

    const complianceData = Object.entries(summary.compliance_distribution || {}).map(([name, value], i) => ({
        name: name.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase()),
        value,
        color: COLORS.compliance[i % COLORS.compliance.length]
    }));

    const trendsChartData = (trends?.data_points || []).map((point, i) => ({
        index: i + 1,
        compliance: point.compliance_score,
        risk: point.risk_score,
        violations: point.violation_count,
        label: `Call ${i + 1}`
    }));

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
            {/* Header */}
            <div className="max-w-7xl mx-auto">
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h1 className="text-3xl font-bold text-white mb-2">📊 Analytics Dashboard</h1>
                        <p className="text-slate-400">
                            Insights from {summary.report_count} analyzed calls
                        </p>
                    </div>
                    <Link
                        href="/"
                        className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg transition"
                    >
                        ← Back
                    </Link>
                </div>

                {/* Summary Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                    <StatCard
                        label="Avg Compliance"
                        value={`${summary.avg_compliance_score}%`}
                        color={summary.avg_compliance_score >= 70 ? "green" : summary.avg_compliance_score >= 50 ? "yellow" : "red"}
                    />
                    <StatCard
                        label="Avg Risk Score"
                        value={summary.avg_risk_score.toString()}
                        color={summary.avg_risk_score <= 30 ? "green" : summary.avg_risk_score <= 60 ? "yellow" : "red"}
                    />
                    <StatCard label="Total Violations" value={summary.total_violations.toString()} color="red" />
                    <StatCard label="Total Calls" value={summary.report_count.toString()} color="blue" />
                </div>

                {/* Charts Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Risk Distribution Pie */}
                    <ChartCard title="Risk Level Distribution">
                        <ResponsiveContainer width="100%" height={280}>
                            <PieChart>
                                <Pie
                                    data={riskData}
                                    cx="50%"
                                    cy="50%"
                                    outerRadius={100}
                                    innerRadius={50}
                                    paddingAngle={3}
                                    dataKey="value"
                                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                    labelLine={false}
                                >
                                    {riskData.map((entry, index) => (
                                        <Cell key={index} fill={entry.color} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{ backgroundColor: "#1e293b", border: "none", borderRadius: "8px" }}
                                    labelStyle={{ color: "#fff" }}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </ChartCard>

                    {/* Intent Distribution Pie */}
                    <ChartCard title="Call Intent Breakdown">
                        <ResponsiveContainer width="100%" height={280}>
                            <PieChart>
                                <Pie
                                    data={intentData}
                                    cx="50%"
                                    cy="50%"
                                    outerRadius={100}
                                    innerRadius={50}
                                    paddingAngle={3}
                                    dataKey="value"
                                    label={({ name, value }) => `${name}: ${value}`}
                                    labelLine={false}
                                >
                                    {intentData.map((entry, index) => (
                                        <Cell key={index} fill={entry.color} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{ backgroundColor: "#1e293b", border: "none", borderRadius: "8px" }}
                                    labelStyle={{ color: "#fff" }}
                                />
                            </PieChart>
                        </ResponsiveContainer>
                    </ChartCard>

                    {/* Compliance Status Bar */}
                    <ChartCard title="Compliance Status">
                        <ResponsiveContainer width="100%" height={280}>
                            <BarChart data={complianceData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                <XAxis type="number" stroke="#9ca3af" />
                                <YAxis dataKey="name" type="category" stroke="#9ca3af" width={100} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: "#1e293b", border: "none", borderRadius: "8px" }}
                                    labelStyle={{ color: "#fff" }}
                                />
                                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                    {complianceData.map((entry, index) => (
                                        <Cell key={index} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </ChartCard>

                    {/* Trends Line Chart */}
                    <ChartCard title="Compliance & Risk Trends">
                        <ResponsiveContainer width="100%" height={280}>
                            <AreaChart data={trendsChartData}>
                                <defs>
                                    <linearGradient id="colorCompliance" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                                    </linearGradient>
                                    <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                <XAxis dataKey="label" stroke="#9ca3af" />
                                <YAxis stroke="#9ca3af" />
                                <Tooltip
                                    contentStyle={{ backgroundColor: "#1e293b", border: "none", borderRadius: "8px" }}
                                    labelStyle={{ color: "#fff" }}
                                />
                                <Legend />
                                <Area
                                    type="monotone"
                                    dataKey="compliance"
                                    stroke="#22c55e"
                                    fillOpacity={1}
                                    fill="url(#colorCompliance)"
                                    name="Compliance Score"
                                />
                                <Area
                                    type="monotone"
                                    dataKey="risk"
                                    stroke="#ef4444"
                                    fillOpacity={1}
                                    fill="url(#colorRisk)"
                                    name="Risk Score"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </ChartCard>
                </div>

                {/* Additional Stats */}
                <div className="mt-8 grid grid-cols-2 md:grid-cols-5 gap-4">
                    <MiniStat label="Min Compliance" value={`${summary.min_compliance_score}%`} />
                    <MiniStat label="Max Compliance" value={`${summary.max_compliance_score}%`} />
                    <MiniStat label="PII Detected" value={summary.total_pii_detected.toString()} />
                    <MiniStat label="Obligations" value={summary.total_obligations.toString()} />
                    <MiniStat label="Total Duration" value={`${Math.round(summary.total_duration_seconds / 60)}m`} />
                </div>

                {/* CSV Export Status */}
                {summary.can_export_csv && (
                    <div className="mt-8 p-4 bg-green-900/30 border border-green-700 rounded-lg">
                        <p className="text-green-400">
                            ✓ CSV export available ({summary.report_count} reports, minimum {summary.min_batch_size} required)
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}

// Helper Components
function StatCard({ label, value, color }: { label: string; value: string; color: string }) {
    const colorClasses = {
        green: "from-green-600 to-green-700",
        yellow: "from-yellow-600 to-yellow-700",
        red: "from-red-600 to-red-700",
        blue: "from-blue-600 to-blue-700"
    };
    return (
        <div className={`bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses] || colorClasses.blue} p-4 rounded-xl`}>
            <div className="text-white/70 text-sm">{label}</div>
            <div className="text-white text-2xl font-bold">{value}</div>
        </div>
    );
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
    return (
        <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4">
            <h3 className="text-white font-semibold mb-4">{title}</h3>
            {children}
        </div>
    );
}

function MiniStat({ label, value }: { label: string; value: string }) {
    return (
        <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-3 text-center">
            <div className="text-slate-400 text-xs">{label}</div>
            <div className="text-white font-bold">{value}</div>
        </div>
    );
}
