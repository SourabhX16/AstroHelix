"use client"

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react"
import { ChevronUp, Play, Square, Loader2 } from "lucide-react"

import { cn } from "@/lib/utils"

export interface TerminalDrawerHandle {
  triggerWithPrompt(prompt: string): void
}

interface TerminalDrawerProps {
  isOpen: boolean
  onToggle: () => void
  className?: string
  onManeuverExecuted?: (data: { position: number[]; velocity: number[]; delta_v: number[] }) => void
}

interface AgentLog {
  id: number
  timestamp: string
  text: string
  color: string // tailwind text color class
}

function formatTime(ts?: number): string {
  const d = ts ? new Date(ts * 1000) : new Date()
  return d.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" })
}

function readable(value: unknown, fallback = ""): string {
  if (typeof value !== "string") return fallback
  const text = value.replace(/[\r\n]+/g, " ").trim()
  if (!text) return fallback
  try {
    const parsed = JSON.parse(text)
    if (parsed && typeof parsed === "object") return "result available"
  } catch {
    // Ordinary prose stays readable; structured payloads are hidden below.
  }
  return text
}

function formatMeters(value: unknown): string {
  const number = Number(value)
  return Number.isFinite(number) ? `${Math.round(number).toLocaleString("en-US")} m` : "unknown miss"
}

function eventToLog(event: Record<string, unknown>, id: number): AgentLog {
  const ts = formatTime(event.timestamp as number | undefined)
  const agent = (event.agent as string) ?? ""
  const type = event.type as string

  switch (type) {
    case "agent_start":
      return { id, timestamp: ts, text: `${agent}: starting analysis...`, color: "text-cyan-400" }
    case "llm_call":
      return { id, timestamp: ts, text: `${agent}: calling LLM (iteration ${event.iteration})...`, color: "text-purple-400" }
    case "tool_calls":
      return {
        id,
        timestamp: ts,
        text: `${agent}: calling tools → ${(event.tools as string[])?.join(", ") ?? ""}`,
        color: "text-yellow-300",
      }
    case "thinking":
      return { id, timestamp: ts, text: `${agent}: ${readable(event.text, "reasoning...")}`, color: "text-gray-400 italic" }
    case "tool_result":
      return {
        id,
        timestamp: ts,
        text: `${agent}.${event.tool}: ${readable(event.summary, "done").slice(0, 120)}`,
        color: "text-emerald-300",
      }
    case "agent_complete":
      return {
        id,
        timestamp: ts,
        text: `${agent}: completed (${event.elapsed_sec}s)`,
        color: "text-green-400",
      }
    case "agent_output":
      return {
        id,
        timestamp: ts,
        text: `${agent}: ${readable(event.content).slice(0, 150)}`,
        color: "text-blue-300",
      }
    case "maneuver_executed":
      return { id, timestamp: ts, text: `${agent}: maneuver applied — updating globe orbit`, color: "text-emerald-400 font-bold" }
    case "maneuver_search": {
      const search = (event.search ?? {}) as Record<string, unknown>
      const candidate = ((search.candidates as Record<string, unknown>[] | undefined) ?? [])[0]
      const direction = candidate?.type ? ` ${candidate.type}` : ""
      const dv = Number(candidate?.delta_v_ms)
      const dvText = Number.isFinite(dv) ? `${dv.toFixed(2)} m/s${direction}` : "no candidate"
      const missBefore = formatMeters(candidate?.miss_before ?? search.miss_distance_m)
      const missAfter = formatMeters(candidate?.miss_after)
      return { id, timestamp: ts, text: `${agent}: searched maneuvers for ${search.secondary_id ?? "threat"}, ${candidate ? `best ${dvText}, miss ${missBefore} → ${missAfter}` : "no improving candidate"}`, color: "text-cyan-300" }
    }
    case "recommended_maneuver": {
      const record = (event.recommended_maneuver ?? {}) as Record<string, unknown>
      const stage = String(event.stage ?? "updated").replaceAll("_", " ")
      const sid = record.secondary_id ? ` for ${record.secondary_id}` : ""
      const verdict = record.verdict ? ` (${record.verdict})` : ""
      const dv = Number(record.delta_v_magnitude_ms ?? record.delta_v_ms)
      const detail = Number.isFinite(dv) ? `, ${dv.toFixed(2)} m/s` : ""
      return { id, timestamp: ts, text: `${agent}: ${stage}${sid}${detail}${verdict}`, color: "text-amber-300" }
    }
    case "iteration_limit":
      return { id, timestamp: ts, text: `${agent}: reached the iteration limit`, color: "text-orange-300" }
    case "info":
      return { id, timestamp: ts, text: readable(event.message, "pipeline information"), color: "text-amber-400" }
    case "pipeline_complete":
      return { id, timestamp: ts, text: "pipeline complete ✓", color: "text-green-500 font-bold" }
    case "error":
      return { id, timestamp: ts, text: `ERROR: ${readable(event.message, "pipeline failed")}`, color: "text-red-400" }
    case "done":
      return { id, timestamp: ts, text: "stream closed", color: "text-gray-500" }
    default:
      return { id, timestamp: ts, text: `${type || "event"}: update received`, color: "text-gray-400" }
  }
}

export const TerminalDrawer = forwardRef<TerminalDrawerHandle, TerminalDrawerProps>(function TerminalDrawer({ isOpen, onToggle, className, onManeuverExecuted }, ref) {
  const [logs, setLogs] = useState<AgentLog[]>([
    { id: 0, timestamp: formatTime(), text: "agent terminal ready — click ▶ to run pipeline", color: "text-gray-500" },
  ])
  const [running, setRunning] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const idRef = useRef(1)

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [logs])

  const startPipeline = useCallback(async (prompt?: string) => {
    if (running) return
    setRunning(true)
    const initMsg = prompt
      ? "auto-triggered by feed data — connecting to agent pipeline..."
      : "connecting to agent pipeline..."
    setLogs([{ id: 0, timestamp: formatTime(), text: initMsg, color: "text-cyan-400" }])
    idRef.current = 1

    const ctrl = new AbortController()
    abortRef.current = ctrl

    try {
      const constraintsResponse = await fetch("/api/constraints", { cache: "no-store", signal: ctrl.signal })
      const constraints = constraintsResponse.ok ? await constraintsResponse.json() : {}
      const params = new URLSearchParams({
        mode: "multi",
        max_dv: String(constraints.max_dv_ms ?? constraints.max_dv ?? 0.35),
        max_burns: String(constraints.max_burns ?? 1),
        axis: String(constraints.axis ?? "along-track"),
        horizon_h: String(constraints.horizon_h ?? 24),
      })
      if (prompt) {
        params.set("prompt", prompt)
      }
      const url = `/api/agent/stream?${params.toString()}`
      const res = await fetch(url, { signal: ctrl.signal })

      if (!res.ok || !res.body) {
        setLogs((prev) => [
          ...prev,
          {
            id: idRef.current++,
            timestamp: formatTime(),
            text: `connection failed (HTTP ${res.status}) — is the agent backend running on :8000?`,
            color: "text-red-400",
          },
        ])
        setRunning(false)
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() ?? ""

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue
          try {
            const event = JSON.parse(line.slice(6))
            const log = eventToLog(event, idRef.current++)
            setLogs((prev) => [...prev, log])

            // Notify parent when the agent executes a maneuver
            if (event.type === "maneuver_executed" && onManeuverExecuted) {
              onManeuverExecuted({ position: event.position, velocity: event.velocity, delta_v: event.delta_v ?? [0, 0, 0] })
            }
          } catch {
            // ignore malformed lines
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setLogs((prev) => [
          ...prev,
          { id: idRef.current++, timestamp: formatTime(), text: `error: ${e}`, color: "text-red-400" },
        ])
      }
    } finally {
      setRunning(false)
      abortRef.current = null
    }
  }, [running, onManeuverExecuted])

  const stopPipeline = useCallback(() => {
    abortRef.current?.abort()
    setLogs((prev) => [
      ...prev,
      { id: idRef.current++, timestamp: formatTime(), text: "pipeline aborted by user", color: "text-orange-400" },
    ])
    setRunning(false)
  }, [])

  useImperativeHandle(ref, () => ({
    triggerWithPrompt(prompt: string) {
      void startPipeline(prompt)
    },
  }), [startPipeline])

  return (
    <div className={cn("pointer-events-auto w-full", className)}>
      <div
        className={cn(
          "overflow-hidden rounded-t-xl border border-border/80 bg-black/85 shadow-2xl transition-[max-height] duration-500 ease-in-out",
          isOpen ? "max-h-72" : "max-h-11"
        )}
      >
        <button
          type="button"
          onClick={onToggle}
          className="flex h-11 w-full items-center justify-between px-4 text-xs font-semibold uppercase tracking-wide text-gray-300 hover:bg-white/5"
        >
          <span className="flex items-center gap-2">
            Agent Terminal
            {running && <Loader2 className="h-3 w-3 animate-spin text-cyan-400" />}
          </span>
          <div className="flex items-center gap-2">
            {!running ? (
              <span
                role="button"
                tabIndex={0}
                onClick={(e) => { e.stopPropagation(); startPipeline() }}
                onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); startPipeline() } }}
                className="flex items-center gap-1 rounded px-2 py-0.5 text-emerald-400 hover:bg-emerald-400/10"
              >
                <Play className="h-3 w-3" /> Run
              </span>
            ) : (
              <span
                role="button"
                tabIndex={0}
                onClick={(e) => { e.stopPropagation(); stopPipeline() }}
                onKeyDown={(e) => { if (e.key === "Enter") { e.stopPropagation(); stopPipeline() } }}
                className="flex items-center gap-1 rounded px-2 py-0.5 text-red-400 hover:bg-red-400/10"
              >
                <Square className="h-3 w-3" /> Stop
              </span>
            )}
            <ChevronUp className={cn("h-4 w-4 transition-transform duration-300", isOpen && "rotate-180")} />
          </div>
        </button>

        <div className="h-60 border-t border-border/70 px-4 py-3">
          <div ref={scrollRef} className="h-full overflow-auto rounded-md bg-black/50 p-3 font-mono text-xs">
            {logs.map((log) => (
              <div key={log.id} className={cn("mb-1 last:mb-0", log.color)}>
                <span className="text-gray-600">[{log.timestamp}]</span> {log.text}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
})
