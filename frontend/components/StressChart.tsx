"use client";

import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
} from "recharts";

interface StressDataPoint {
    time: string;
    stress: number;
}

interface StressChartProps {
    data: StressDataPoint[];
}

export default function StressChart({ data }: StressChartProps) {
    if (!data || data.length === 0) {
        return (
            <div className="w-full h-96 flex items-center justify-center bg-gray-900 rounded-lg border border-gray-700">
                <p className="text-gray-400">No stress timeline data available</p>
            </div>
        );
    }

    // Find max stress level for reference line
    const maxStress = Math.max(...data.map((d) => d.stress));
    const avgStress = data.reduce((sum, d) => sum + d.stress, 0) / data.length;

    return (
        <div className="w-full bg-gray-900 rounded-lg border border-gray-700 p-4">
            <h3 className="text-lg font-semibold text-gray-100 mb-4">
                Stress Timeline
            </h3>
            <ResponsiveContainer width="100%" height={300}>
                <LineChart data={data} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis
                        dataKey="time"
                        stroke="#6B7280"
                        style={{ fontSize: "0.875rem" }}
                    />
                    <YAxis
                        stroke="#6B7280"
                        domain={[0, maxStress * 1.1]}
                        style={{ fontSize: "0.875rem" }}
                    />
                    <Tooltip
                        contentStyle={{
                            backgroundColor: "#111827",
                            border: "1px solid #374151",
                            borderRadius: "0.5rem",
                        }}
                        labelStyle={{ color: "#E5E7EB" }}
                        formatter={(value: number) => [value.toFixed(1), "Stress Level"]}
                    />
                    <ReferenceLine
                        y={avgStress}
                        stroke="#60A5FA"
                        strokeDasharray="5 5"
                        label={{
                            value: `Average: ${avgStress.toFixed(1)}`,
                            position: "right",
                            fill: "#60A5FA",
                            fontSize: 12,
                        }}
                    />
                    <Line
                        type="monotone"
                        dataKey="stress"
                        stroke="#EF4444"
                        dot={{ fill: "#EF4444", r: 4 }}
                        activeDot={{ r: 6 }}
                        strokeWidth={2}
                        isAnimationActive={true}
                    />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}
