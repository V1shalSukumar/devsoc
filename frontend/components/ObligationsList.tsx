"use client";

interface Obligation {
    id?: string;
    title: string;
    description: string;
    speaker?: string;
    quote?: string;
    timestamp?: string;
    status?: "pending" | "completed" | "failed";
}

interface ObligationsListProps {
    obligations: Obligation[] | string[];
}

const statusColors: Record<string, string> = {
    pending: "bg-yellow-900 text-yellow-100",
    completed: "bg-green-900 text-green-100",
    failed: "bg-red-900 text-red-100",
};

export default function ObligationsList({ obligations }: ObligationsListProps) {
    if (!obligations || obligations.length === 0) {
        return (
            <div className="bg-gray-900 rounded-lg border border-gray-700 p-6 text-center">
                <p className="text-gray-400">No obligations recorded</p>
            </div>
        );
    }

    // Normalize obligations to objects
    const normalizedObligations: Obligation[] = obligations.map((o) => {
        if (typeof o === "string") {
            return {
                title: "Commitment",
                description: o,
                status: "pending",
            };
        }
        return o;
    });

    return (
        <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
            <div className="p-4 border-b border-gray-700 bg-gray-800">
                <h3 className="text-lg font-semibold text-gray-100">
                    Obligations ({normalizedObligations.length})
                </h3>
            </div>
            <div className="divide-y divide-gray-700 max-h-[400px] overflow-y-auto">
                {normalizedObligations.map((obligation, idx) => (
                    <div key={idx} className="p-4 hover:bg-gray-800/50 transition-colors">
                        <div className="flex justify-between items-start mb-2">
                            <h4 className="font-semibold text-gray-100">
                                {obligation.title}
                            </h4>
                            {obligation.status && (
                                <span
                                    className={`px-2 py-1 rounded text-xs font-medium ${statusColors[obligation.status]
                                        }`}
                                >
                                    {obligation.status.toUpperCase()}
                                </span>
                            )}
                        </div>

                        <p className="text-gray-300 text-sm mb-3">
                            {obligation.description}
                        </p>

                        {obligation.speaker && (
                            <div className="mb-2 text-xs text-gray-400">
                                <p>
                                    <span className="font-semibold">Speaker:</span>{" "}
                                    {obligation.speaker}
                                </p>
                            </div>
                        )}

                        {obligation.timestamp && (
                            <p className="text-xs text-gray-400">
                                Time: {obligation.timestamp}
                            </p>
                        )}

                        {obligation.quote && (
                            <div className="mt-3 p-3 bg-gray-800/50 rounded border-l-2 border-blue-500">
                                <p className="text-xs text-gray-300 italic">
                                    "{obligation.quote}"
                                </p>
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
