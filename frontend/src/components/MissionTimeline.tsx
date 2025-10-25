import { useMemo } from "react";
import type { BackgroundEvent, TranscriptSegment } from "../api/types";
import "./MissionTimeline.css";

interface MissionTimelineProps {
  transcript: TranscriptSegment[];
  events: BackgroundEvent[];
  duration?: number | null;
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
          {transcript.map((segment) => {
            const startPercent = (segment.start_s / totalDuration) * 100;
            const widthPercent =
              ((segment.end_s - segment.start_s) / totalDuration) * 100;
            const safeWidth = Math.max(Math.min(widthPercent, 100 - startPercent), 4);
            return (
              <div
                key={`${segment.start_s}-${segment.end_s}-${segment.speaker}`}
                className="timeline__block timeline__block--transcript"
                style={{
                  left: `${startPercent}%`,
                  width: `${safeWidth}%`,
                }}
              >
                <span className="timeline__block-speaker">{segment.speaker}</span>
                <span className="timeline__block-text">{segment.text}</span>
              </div>
            );
          })}
        </div>

        <div className="timeline__lane timeline__lane--events">
          {events.map((event, index) => {
            const startPercent = (event.start_s / totalDuration) * 100;
            const widthPercent = ((event.end_s - event.start_s) / totalDuration) * 100;
            const safeWidth = Math.max(Math.min(widthPercent, 100 - startPercent), 2);
            return (
              <div
                key={`${event.label}-${index}`}
                className={`timeline__block timeline__block--event timeline__block--event-${sanitizeLabel(
                  event.label,
                )}`}
                style={{
                  left: `${startPercent}%`,
                  width: `${safeWidth}%`,
                  opacity: event.probability
                    ? Math.min(Math.max(event.probability, 0.2), 0.95)
                    : 0.6,
                }}
              >
                <span className="timeline__block-label">
                  {event.label}{" "}
                  {event.probability !== undefined && event.probability !== null
                    ? `(${Math.round(event.probability * 100)}%)`
                    : ""}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="timeline__intel">
        <div className="timeline__intel-column">
          <h4>Transcript Log</h4>
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

        <div className="timeline__intel-column">
          <h4>Background Events</h4>
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
