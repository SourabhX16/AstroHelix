import type { Metadata } from "next"

import "./globals.css"

export const metadata: Metadata = {
  title: "Astro Helix",
  description: "Autonomous debris avoidance dashboard",
  icons: {
    icon: [
      { url: "/favicon-32.png", sizes: "32x32" },
      { url: "/favicon-192.png", sizes: "192x192" },
    ],
    apple: "/apple-touch-icon.png",
  },
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body suppressHydrationWarning>{children}</body>
    </html>
  )
}
