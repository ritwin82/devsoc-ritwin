"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchReports } from "@/lib/api";
import AudioUpload from "@/components/AudioUpload";
import type { Report } from "@/lib/api";

export default function Home() {
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    const loadReports = async () => {
      try {
        const data = await fetchReports();
        setReports(data);
      } catch (err) {
        console.error("Failed to load reports:", err);
      } finally {
        setLoading(false);
      }
    };

    loadReports();
  }, []);

  const handleAudioAnalyzed = async (report: Report) => {
    // Report already analyzed by AudioUpload component — just refresh the list
    setIsAnalyzing(false);
    try {
      const data = await fetchReports();
      setReports(data);
    } catch (err) {
      console.error("Failed to refresh reports:", err);
    }
  };

  const avgRiskScore =
    reports.length > 0
      ? (
        reports.reduce((sum, r) => sum + (r.risk_score || 0), 0) / reports.length
      ).toFixed(1)
      : "0.0";

  const highRiskCount = reports.filter(
    (r) => (r.risk_score || 0) >= 7
  ).length;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Hero Section */}
      <div className="bg-gradient-to-b from-gray-900 to-gray-950 border-b border-gray-800">
        <div className="max-w-6xl mx-auto px-4 py-16 sm:py-24">
          <div className="text-center mb-12">
            <h1 className="text-5xl sm:text-6xl font-bold mb-4">
              Financial Audio Intelligence
            </h1>
            <p className="text-xl text-gray-400 mb-8">
              Compliance analysis for financial service call recordings
            </p>
          </div>

          {/* Upload Card */}
          <div className="bg-gray-900 rounded-lg border border-gray-700 p-8 max-w-2xl mx-auto">
            <h2 className="text-2xl font-semibold mb-6">Upload & Analyze Call</h2>
            <AudioUpload onAnalyzed={handleAudioAnalyzed} />
            {isAnalyzing && (
              <p className="mt-4 text-center text-blue-400">
                Analyzing your call...
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Dashboard Section */}
      <div className="max-w-6xl mx-auto px-4 py-16">
        {/* Stats Grid */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold mb-6">Overview</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Total Calls */}
            <div className="bg-gray-900 rounded-lg border border-gray-700 p-6">
              <p className="text-sm text-gray-400 mb-2">Total Calls Analyzed</p>
              <p className="text-4xl font-bold">{reports.length}</p>
              {!loading && reports.length === 0 && (
                <p className="text-xs text-gray-500 mt-2">
                  Upload your first call to get started
                </p>
              )}
            </div>

            {/* Average Risk Score */}
            <div className="bg-gray-900 rounded-lg border border-gray-700 p-6">
              <p className="text-sm text-gray-400 mb-2">Average Risk Score</p>
              <p className="text-4xl font-bold">{avgRiskScore}</p>
              <p className="text-xs text-gray-500 mt-2">
                {reports.length === 0 ? "No data yet" : "out of 10"}
              </p>
            </div>

            {/* High Risk Calls */}
            <div className="bg-red-900/20 border border-red-700 rounded-lg p-6">
              <p className="text-sm text-red-200 mb-2">High Risk Calls</p>
              <p className="text-4xl font-bold text-red-100">{highRiskCount}</p>
              {highRiskCount > 0 && (
                <p className="text-xs text-red-300 mt-2">Requires attention</p>
              )}
            </div>

            {/* View Reports Button */}
            <Link href="/reports">
              <div className="bg-blue-900/20 border border-blue-700 rounded-lg p-6 hover:bg-blue-900/30 transition-colors cursor-pointer h-full flex flex-col justify-center">
                <p className="text-sm text-blue-200 mb-2">All Reports</p>
                <p className="text-2xl font-bold text-blue-100">
                  View →
                </p>
                <p className="text-xs text-blue-300 mt-2">
                  Detailed analytics
                </p>
              </div>
            </Link>
          </div>
        </div>

        {/* Recent Reports Preview */}
        {!loading && reports.length > 0 && (
          <div>
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-3xl font-bold">Recent Calls</h2>
              <Link href="/reports" className="text-blue-400 hover:text-blue-300">
                View all →
              </Link>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {reports.slice(0, 3).map((report, idx) => {
                const getRiskBadgeColor = (score: number) => {
                  if (score >= 7) return "bg-red-900 text-red-100";
                  if (score >= 4) return "bg-yellow-900 text-yellow-100";
                  return "bg-green-900 text-green-100";
                };

                const riskScore = report.risk_score || 0;
                const filename = report.filename || "Unknown Call";
                const timestamp = report.timestamp || new Date().toISOString();
                const reportId = report.id || `report-${idx}`;

                return (
                  <Link key={reportId} href={`/reports/${reportId}`}>
                    <div className="p-4 rounded-lg border border-gray-700 bg-gray-900 hover:bg-gray-800 transition-colors">
                      <div className="flex justify-between items-start mb-2">
                        <h3 className="font-semibold text-gray-100 truncate">
                          {filename}
                        </h3>
                        <span
                          className={`px-2 py-1 rounded text-xs font-medium whitespace-nowrap ml-2 ${getRiskBadgeColor(
                            riskScore
                          )}`}
                        >
                          {riskScore.toFixed(1)}
                        </span>
                      </div>
                      <p className="text-xs text-gray-400">
                        {new Date(timestamp).toLocaleDateString()}
                      </p>
                    </div>
                  </Link>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
