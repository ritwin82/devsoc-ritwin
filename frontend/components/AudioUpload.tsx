"use client";

import React, { useState } from "react";
import { analyzeAudio } from "@/lib/api";
import type { Report } from "@/lib/api";

interface AudioUploadProps {
  onAnalyzed?: (report: Report) => void;
}

export default function AudioUpload({ onAnalyzed }: AudioUploadProps) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null;

    if (selectedFile) {
      if (selectedFile.type.startsWith("audio/")) {
        setFile(selectedFile);
        setMessage(null);
      } else {
        setMessage({
          type: "error",
          text: "Please select a valid audio file (MP3, WAV, M4A, etc.)",
        });
        setFile(null);
      }
    }
  };

  const uploadAudio = async () => {
    if (!file) {
      setMessage({ type: "error", text: "Please select an audio file first" });
      return;
    }

    setLoading(true);
    setMessage(null);

    try {
      const result = await analyzeAudio(file);

      if (result) {
        setMessage({
          type: "success",
          text: `✓ Successfully analyzed: ${result.filename}`,
        });
        setFile(null);

        // Call callback with the full report
        if (onAnalyzed) {
          onAnalyzed(result);
        }

        // Reset message after 3 seconds
        setTimeout(() => {
          setMessage(null);
        }, 3000);
      } else {
        setMessage({
          type: "error",
          text: "Failed to analyze audio. Please try again.",
        });
      }
    } catch (error) {
      setMessage({
        type: "error",
        text: "Error uploading file. Please check your backend connection.",
      });
      console.error("Upload error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full">
      <div className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
        <div className="mb-4">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            stroke="currentColor"
            fill="none"
            viewBox="0 0 48 48"
          >
            <path
              d="M28 8H12a4 4 0 00-4 4v20a4 4 0 004 4h24a4 4 0 004-4V20m-6-8l-6-6m0 0l-6 6m6-6v16"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>

        <label className="cursor-pointer">
          <span className="text-blue-400 hover:text-blue-300 font-medium">
            Click to upload
          </span>
          <span className="text-gray-400"> or drag and drop</span>
          <input
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            className="hidden"
            disabled={loading}
          />
        </label>

        <p className="text-xs text-gray-500 mt-2">
          MP3, WAV, M4A, OGG, FLAC, WebM up to 100MB
        </p>

        {file && (
          <div className="mt-4">
            <p className="text-gray-300 font-medium">Selected: {file.name}</p>
            <p className="text-sm text-gray-400">
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
        )}
      </div>

      {/* Message */}
      {message && (
        <div
          className={`mt-4 p-3 rounded-lg text-sm ${message.type === "success"
              ? "bg-green-900/20 border border-green-700 text-green-200"
              : "bg-red-900/20 border border-red-700 text-red-200"
            }`}
        >
          {message.text}
        </div>
      )}

      {/* Upload Button */}
      <button
        onClick={uploadAudio}
        disabled={!file || loading}
        className={`mt-4 w-full py-3 px-4 rounded-lg font-medium transition-colors ${!file || loading
            ? "bg-gray-700 text-gray-400 cursor-not-allowed"
            : "bg-blue-600 hover:bg-blue-700 text-white"
          }`}
      >
        {loading ? "Analyzing..." : "Upload & Analyze"}
      </button>
    </div>
  );
}
