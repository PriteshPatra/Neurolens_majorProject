import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";
import { SettingsProvider } from "@/context/SettingsContext";

export const metadata: Metadata = {
  title: "Neurolens | Stress Detection",
  description: "Real-Time Stress Detection & Mitigation System",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <SettingsProvider>
          <Navbar />
          <main className="container">
            {children}
          </main>
        </SettingsProvider>
      </body>
    </html>
  );
}
