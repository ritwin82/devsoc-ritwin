import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Financial Audio Intelligence",
  description: "Compliance analysis for financial service call recordings",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gray-950 text-gray-100`}
      >
        {/* Navigation Header */}
        <nav className="bg-gray-900 border-b border-gray-800">
          <div className="max-w-6xl mx-auto px-4 py-4 flex justify-between items-center">
            <Link href="/" className="text-xl font-bold text-blue-400 hover:text-blue-300">
              📞 FinAI
            </Link>
            <div className="flex items-center gap-6">
              <Link
                href="/"
                className="text-gray-300 hover:text-white transition-colors"
              >
                Home
              </Link>
              <Link
                href="/reports"
                className="text-gray-300 hover:text-white transition-colors"
              >
                Reports
              </Link>
              <Link
                href="/realtime"
                className="text-gray-300 hover:text-white transition-colors"
              >
                Real-Time
              </Link>
              <Link
                href="/query"
                className="text-gray-300 hover:text-white transition-colors"
              >
                Query Bot
              </Link>
            </div>
          </div>
        </nav>

        {/* Content */}
        {children}
      </body>
    </html>
  );
}
