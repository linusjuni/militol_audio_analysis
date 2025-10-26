import { useMemo, useState } from "react";
import type { CSSProperties, KeyboardEvent } from "react";
import type { BackgroundEvent, TranscriptSegment } from "../api/types";
import "./MissionTimeline.css";

interface MissionTimelineProps {
  transcript: TranscriptSegment[];
  events: BackgroundEvent[];
  duration?: number | null;
}

type TooltipType = "transcript" | "event";

interface TooltipData {
  key: string;
  type: TooltipType;
  header: string;
  time: string;
  body?: string;
  meta?: string;
  pinned: boolean;
}

type TooltipInput = Omit<TooltipData, "pinned">;

const MIN_TRANSCRIPT_WIDTH = 3;
const MIN_EVENT_WIDTH = 2.2;
const TRANSCRIPT_GAP = 0.6;
const EVENT_GAP = 0.4;

interface BlockLayout<T> {
  item: T;
  key: string;
  actualStartPct: number;
  actualEndPct: number;
  collapsedStartPct: number;
  collapsedEndPct: number;
}

interface PrepareBlocksOptions<T> {
  minWidthPct: number;
  gapPct: number;
  createKey: (item: T, index: number) => string;
}

export function MissionTimeline({ transcript, events, duration }: MissionTimelineProps) {
  const totalDuration = useMemo(() => {
    const maxFromTranscript = transcript.reduce(
      (acc, seg) => Math.max(acc, seg.end_s ?? 0),
      0,
    );
    const maxFromEvents = events.reduce((acc, ev) => Math.max(acc, ev.end_s ?? 0), 0);
    return duration && duration > 0
      ? Math.max(duration, maxFromTranscript, maxFromEvents)
      : Math.max(maxFromTranscript, maxFromEvents, 1);
  }, [transcript, events, duration]);

  const scaleMarks = useMemo(() => {
    const marks = [];
    const increment = totalDuration > 120 ? 30 : totalDuration > 60 ? 15 : 5;
    for (let t = 0; t <= totalDuration + 0.01; t += increment) {
      marks.push({ t, label: formatSeconds(t) });
    }
    return marks;
  }, [totalDuration]);

  const [tooltip, setTooltip] = useState<TooltipData | null>(null);

  const transcriptBlocks = useMemo(
    () =>
      prepareBlocks(transcript, totalDuration, {
        minWidthPct: MIN_TRANSCRIPT_WIDTH,
        gapPct: TRANSCRIPT_GAP,
        createKey: (segment, index) =>
          `transcript-${segment.start_s}-${segment.end_s}-${segment.speaker ?? index}`,
      }),
    [transcript, totalDuration],
  );

  const eventBlocks = useMemo(
    () =>
      prepareBlocks(events, totalDuration, {
        minWidthPct: MIN_EVENT_WIDTH,
        gapPct: EVENT_GAP,
        createKey: (event, index) => `event-${event.label}-${index}-${event.start_s}`,
      }),
    [events, totalDuration],
  );

  const handleMouseEnter = (payload: TooltipInput) => {
    setTooltip((current) => {
      if (current?.pinned && current.key !== payload.key) {
        return current;
      }
      if (current?.key === payload.key && current.pinned) {
        return current;
      }
      return { ...payload, pinned: false };
    });
  };

  const handleMouseLeave = (key: string) => {
    setTooltip((current) => {
      if (current?.key === key && !current.pinned) {
        return null;
      }
      return current;
    });
  };

  const handleBlockClick = (payload: TooltipInput) => {
    setTooltip((current) => {
      if (current?.key === payload.key) {
        return { ...current, pinned: !current.pinned };
      }
      return { ...payload, pinned: true };
    });
  };

  const handleBlockKeyDown = (event: KeyboardEvent<HTMLDivElement>, payload: TooltipInput) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      handleBlockClick(payload);
    } else if (event.key === "Escape" && tooltip?.key === payload.key) {
      event.preventDefault();
      setTooltip(null);
    }
  };

  return (
    <div className="timeline">
      <div className="timeline__tracks">
        <div className="timeline__scale">
          {scaleMarks.map((mark) => (
            <span
              key={mark.t}
              className="timeline__scale-mark"
              style={{ left: `${(mark.t / totalDuration) * 100}%` }}
            >
              {mark.label}
            </span>
          ))}
        </div>

        <div className="timeline__lane timeline__lane--transcript">
          {transcriptBlocks.map((block) => {
            const {
              item: segment,
              key: blockKey,
              actualStartPct,
              actualEndPct,
              collapsedStartPct,
              collapsedEndPct,
            } = block;
            const displaySpeaker = segment.speaker || "Speaker";
            const transcriptBody = segment.text?.trim();
            const summary = transcriptBody ? truncateText(transcriptBody, 70) : "No transcript";
            const payload: TooltipInput = {
              key: blockKey,
              type: "transcript",
              header: displaySpeaker,
              time: formatRange(segment.start_s, segment.end_s),
              body: transcriptBody,
            };
            const isActive = tooltip?.key === blockKey;
            const isPinned = isActive && tooltip?.pinned;
            const collapsedWidthPct = Math.max(collapsedEndPct - collapsedStartPct, MIN_TRANSCRIPT_WIDTH);
            const actualWidthPct = Math.max(actualEndPct - actualStartPct, MIN_TRANSCRIPT_WIDTH);
            const blockStyle: CSSProperties = {
              "--timeline-block-left": `${collapsedStartPct}%`,
              "--timeline-block-width": `${collapsedWidthPct}%`,
              "--timeline-block-actual-left": `${actualStartPct}%`,
              "--timeline-block-actual-width": `${actualWidthPct}%`,
            } as CSSProperties;
            return (
              <div
                key={blockKey}
                className={`timeline__block timeline__block--transcript timeline__block--interactive${
                  isActive ? " timeline__block--active timeline__block--expanded" : ""
                }${isPinned ? " timeline__block--pinned" : ""}`}
                tabIndex={0}
                role="button"
                aria-label={`Transcript segment by ${displaySpeaker} from ${payload.time}`}
                aria-pressed={isPinned}
                style={blockStyle}
                title={transcriptBody ? `${displaySpeaker}: ${transcriptBody}` : displaySpeaker}
                onMouseEnter={() => handleMouseEnter(payload)}
                onMouseLeave={() => handleMouseLeave(blockKey)}
                onFocus={() => handleMouseEnter(payload)}
                onBlur={() => handleMouseLeave(blockKey)}
                onClick={() => handleBlockClick(payload)}
                onKeyDown={(keyboardEvent) => handleBlockKeyDown(keyboardEvent, payload)}
              >
                <div className="timeline__block-strip" aria-hidden="true" />
                <div className="timeline__block-main">
                  <div className="timeline__block-row">
                    <span className="timeline__block-title">{displaySpeaker}</span>
                    <span className="timeline__block-meta">{payload.time}</span>
                  </div>
                  <span className="timeline__block-summary">{summary}</span>
                </div>
                {isActive && tooltip && (
                  <div className="timeline__tooltip timeline__tooltip--transcript">
                    <span className="timeline__tooltip-time">{tooltip.time}</span>
                    <strong className="timeline__tooltip-title">
                      {tooltip.header}
                    </strong>
                    {tooltip.body && (
                      <p className="timeline__tooltip-body">{tooltip.body}</p>
                    )}
                    <span className="timeline__tooltip-hint">
                      {tooltip.pinned ? "Click to close" : "Click to pin"}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="timeline__lane timeline__lane--events">
          {eventBlocks.map((block) => {
            const {
              item: event,
              key: blockKey,
              actualStartPct,
              actualEndPct,
              collapsedStartPct,
              collapsedEndPct,
            } = block;
            const probability =
              event.probability !== undefined && event.probability !== null
                ? Math.round(event.probability * 100)
                : null;
            const summary =
              probability !== null ? `${probability}% confidence` : "Confidence n/a";
            const blockOpacity =
              event.probability !== undefined && event.probability !== null
                ? Math.min(Math.max(event.probability, 0.2), 0.95)
                : 0.6;
            const payload: TooltipInput = {
              key: blockKey,
              type: "event",
              header: event.label,
              time: formatRange(event.start_s, event.end_s),
              meta:
                probability !== null
                  ? `${probability}% confidence`
                  : "Confidence not available",
            };
            const isActive = tooltip?.key === blockKey;
            const isPinned = isActive && tooltip?.pinned;
            const effectiveOpacity = isActive || isPinned ? 1 : blockOpacity;
            const collapsedWidthPct = Math.max(collapsedEndPct - collapsedStartPct, MIN_EVENT_WIDTH);
            const actualWidthPct = Math.max(actualEndPct - actualStartPct, MIN_EVENT_WIDTH);
            const blockStyle: CSSProperties = {
              "--timeline-block-left": `${collapsedStartPct}%`,
              "--timeline-block-width": `${collapsedWidthPct}%`,
              "--timeline-block-actual-left": `${actualStartPct}%`,
              "--timeline-block-actual-width": `${actualWidthPct}%`,
              opacity: effectiveOpacity,
            } as CSSProperties;
            return (
              <div
                key={blockKey}
                className={`timeline__block timeline__block--event timeline__block--interactive timeline__block--event-${sanitizeLabel(
                  event.label,
                )}${isActive ? " timeline__block--active timeline__block--expanded" : ""}${
                  isPinned ? " timeline__block--pinned" : ""
                }`}
                tabIndex={0}
                role="button"
                aria-label={`Background event ${event.label} at ${payload.time}`}
                aria-pressed={isPinned}
                style={blockStyle}
                title={`${event.label}${
                  probability !== null ? ` (${probability}% confidence)` : ""
                }`}
                onMouseEnter={() => handleMouseEnter(payload)}
                onMouseLeave={() => handleMouseLeave(blockKey)}
                onFocus={() => handleMouseEnter(payload)}
                onBlur={() => handleMouseLeave(blockKey)}
                onClick={() => handleBlockClick(payload)}
                onKeyDown={(keyboardEvent) => handleBlockKeyDown(keyboardEvent, payload)}
              >
                <div className="timeline__block-strip" aria-hidden="true" />
                <div className="timeline__block-main">
                  <div className="timeline__block-row">
                    <span className="timeline__block-title">{event.label}</span>
                    <span className="timeline__block-meta">{payload.time}</span>
                  </div>
                  <span className="timeline__block-summary">{summary}</span>
                </div>
                {isActive && tooltip && (
                  <div className="timeline__tooltip timeline__tooltip--event">
                    <span className="timeline__tooltip-time">{tooltip.time}</span>
                    <strong className="timeline__tooltip-title">
                      {tooltip.header}
                    </strong>
                    {tooltip.meta && (
                      <span className="timeline__tooltip-meta">{tooltip.meta}</span>
                    )}
                    <span className="timeline__tooltip-hint">
                      {tooltip.pinned ? "Click to close" : "Click to pin"}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="timeline__intel">
        <div className="timeline__intel-column">
          <h4>Transcript Log</h4>
          <div className="timeline__intel-body">
            <ul>
              {transcript.map((segment) => (
                <li key={`log-${segment.start_s}`}>
                  <span className="timeline__intel-time">{formatRange(segment.start_s, segment.end_s)}</span>
                  <strong>{segment.speaker}</strong>
                  <span>{segment.text}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="timeline__intel-column">
          <h4>Background Events</h4>
          <div className="timeline__intel-body">
            <ul>
              {events.map((event, idx) => (
                <li key={`ev-${event.label}-${idx}`}>
                  <span className="timeline__intel-time">{formatRange(event.start_s, event.end_s)}</span>
                  <strong>{event.label}</strong>
                  {event.probability !== undefined && event.probability !== null ? (
                    <span>{Math.round(event.probability * 100)}% confidence</span>
                  ) : (
                    <span>confidence n/a</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function sanitizeLabel(label: string): string {
  return label.toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

function formatSeconds(seconds: number): string {
  const whole = Math.floor(seconds);
  const mins = Math.floor(whole / 60);
  const secs = whole % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function formatRange(start: number, end: number): string {
  const safeEnd = end >= start ? end : start;
  const formattedStart = formatSeconds(start);
  const formattedEnd = formatSeconds(safeEnd);
  if (Math.abs(safeEnd - start) < 0.3) {
    return `${formattedStart}s`;
  }
  return `${formattedStart}s – ${formattedEnd}s`;
}

function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 3)}...`;
}

function prepareBlocks<T extends { start_s: number; end_s?: number | null }>(
  items: readonly T[],
  totalDuration: number,
  options: PrepareBlocksOptions<T>,
): BlockLayout<T>[] {
  if (!items.length) {
    return [];
  }

  const safeTotal = totalDuration && totalDuration > 0 ? totalDuration : 1;
  const minWidth = options.minWidthPct;

  type InternalBlockLayout<TItem> = BlockLayout<TItem> & { originalIndex: number };

  const base: InternalBlockLayout<T>[] = items.map((item, index) => {
    const startSeconds = Math.max(0, item.start_s ?? 0);
    const rawEndSeconds =
      item.end_s !== undefined && item.end_s !== null ? item.end_s : item.start_s;
    const safeEndSeconds = Math.max(rawEndSeconds, startSeconds);
    const actualStartPct = clampToPercent((startSeconds / safeTotal) * 100);
    const rawEndPct = clampToPercent((safeEndSeconds / safeTotal) * 100);
    const ensuredEndPct = clampToPercent(Math.max(actualStartPct + minWidth, rawEndPct));
    return {
      item,
      key: options.createKey(item, index),
      actualStartPct,
      actualEndPct: ensuredEndPct,
      collapsedStartPct: actualStartPct,
      collapsedEndPct: ensuredEndPct,
      originalIndex: index,
    };
  });

  const sorted = [...base].sort((a, b) => {
    if (a.actualStartPct !== b.actualStartPct) {
      return a.actualStartPct - b.actualStartPct;
    }
    if (a.actualEndPct !== b.actualEndPct) {
      return a.actualEndPct - b.actualEndPct;
    }
    return a.originalIndex - b.originalIndex;
  });

  for (let i = 0; i < sorted.length - 1; i += 1) {
    const current = sorted[i];
    const next = sorted[i + 1];
    if (current.actualEndPct <= next.actualStartPct) {
      continue;
    }
    const overlapStart = next.actualStartPct;
    const overlapEnd = Math.min(current.actualEndPct, next.actualEndPct);
    if (overlapEnd <= overlapStart) {
      continue;
    }
    const splitPoint = overlapStart + (overlapEnd - overlapStart) / 2;
    current.collapsedEndPct = clamp(
      splitPoint,
      current.collapsedStartPct + minWidth,
      current.actualEndPct,
    );
    const nextMaxStart = Math.max(next.actualStartPct, next.actualEndPct - minWidth);
    next.collapsedStartPct = clamp(splitPoint, next.actualStartPct, nextMaxStart);
  }

  for (let i = 1; i < sorted.length; i += 1) {
    const prev = sorted[i - 1];
    const current = sorted[i];
    const requiredStart = prev.collapsedEndPct + options.gapPct;
    const maxStart = Math.max(current.actualStartPct, current.actualEndPct - minWidth);
    current.collapsedStartPct = clamp(
      Math.max(current.collapsedStartPct, requiredStart),
      current.actualStartPct,
      maxStart,
    );
    current.collapsedEndPct = clamp(
      current.collapsedEndPct,
      current.collapsedStartPct + minWidth,
      current.actualEndPct,
    );
  }

  return sorted
    .sort((a, b) => a.originalIndex - b.originalIndex)
    .map(({ originalIndex: _unused, ...rest }) => rest);
}

function clamp(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) {
    return min;
  }
  if (max < min) {
    return min;
  }
  return Math.min(Math.max(value, min), max);
}

function clampToPercent(value: number): number {
  return clamp(value, 0, 100);
}
