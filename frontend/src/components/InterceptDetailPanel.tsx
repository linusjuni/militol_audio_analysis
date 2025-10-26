import { useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import type { InterceptDetail } from "../api/types";
import { MissionTimeline } from "./MissionTimeline";
import "./InterceptDetailPanel.css";
import { stripFileExtension } from "../utils/text";

interface DetailPanelProps {
  detail: InterceptDetail | null;
  loading: boolean;
  error: string | null;
  interceptId: string | null;
  onDelete?: (interceptId: string) => void;
  onRerun?: (interceptId: string) => void;
  busyId?: string | null;
}

export function InterceptOverviewPanel({
  detail,
  loading,
  error,
  interceptId,
  onDelete,
  onRerun,
  busyId,
}: DetailPanelProps) {
  if (!interceptId) {
    return (
      <div className="detail-overview detail-overview--empty">
        <span>Select an intercept to open analysis.</span>
      </div>
    );
  }

  if (loading && !detail) {
    return (
      <div className="detail-overview detail-overview--loading">
        <span>Preparing intercept intelligence…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="detail-overview detail-overview--error">
        <strong>Unable to load intercept.</strong>
        <span>{error}</span>
      </div>
    );
  }

  if (!detail) {
    return null;
  }

  const { meta } = detail;
  const displayTitle = stripFileExtension(meta.title) || meta.title;
  const startedAt = new Date(meta.created_at).toLocaleString();
  const audioHref = computeAudioUrl(detail, interceptId);
  const tags = meta.tags ?? [];
  const topEvents = (detail.background_events ?? [])
    .map((event) => ({
      ...event,
      probability: event.probability ?? null,
    }))
    .sort((a, b) => (b.probability ?? 0) - (a.probability ?? 0))
    .slice(0, 3);

  return (
    <div className="detail-overview">
      <header className="detail-overview__header">
        <div>
          <span className="detail-overview__eyebrow">STATUS · {meta.status.toUpperCase()}</span>
          <h2>{displayTitle}</h2>
          <div className="detail-overview__meta">
            <span>Captured · {startedAt}</span>
            {meta.duration_s ? (
              <span>Duration · {formatSeconds(meta.duration_s)}</span>
            ) : null}
          </div>
        </div>
        {meta.priority || tags.length > 0 ? (
          <div className="detail-overview__tags">
            {meta.priority ? (
              <span className="detail-overview__priority">{meta.priority}</span>
            ) : null}
            {tags.map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
          </div>
        ) : null}
      </header>

      {meta.executive_summary ? (
        <p className="detail-overview__summary">{meta.executive_summary}</p>
      ) : null}

      {audioHref ? (
        <div className="detail-overview__audio">
          <audio controls src={audioHref}>
            <track kind="captions" />
          </audio>
        </div>
      ) : null}

      <div className="detail-overview__grid">
        <div className="detail-overview__intel">
          <h4>Signal Highlights</h4>
          {topEvents.length === 0 ? (
            <span className="detail-overview__intel-empty">No background detections yet.</span>
          ) : (
            <ul>
              {topEvents.map((event, index) => (
                <li key={`${event.label}-${index}`}>
                  <span className="detail-overview__intel-label">{event.label}</span>
                  <span className="detail-overview__intel-time">
                    {formatRange(event.start_s, event.end_s)}
                  </span>
                  <span className="detail-overview__intel-prob">
                    {event.probability !== null ? `${Math.round(event.probability * 100)}%` : "n/a"}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="detail-overview__actions">
          <h4>Console Actions</h4>
          <button
            type="button"
            disabled={
              !onRerun ||
              !interceptId ||
              meta.status === "processing" ||
              (busyId !== null && busyId === interceptId)
            }
            onClick={() => interceptId && onRerun?.(interceptId)}
          >
            {meta.status === "processing" ? "Pipeline Running" : "Re-run Pipeline"}
          </button>
          <button
            type="button"
            disabled={!onDelete || !interceptId || (busyId !== null && busyId === interceptId)}
            onClick={() => interceptId && onDelete?.(interceptId)}
          >
            Remove Intercept
          </button>
        </div>
      </div>
    </div>
  );
}

export function InterceptDeepDivePanel({
  detail,
  loading,
  error,
  interceptId,
}: DetailPanelProps) {
  const [operatorNotes, setOperatorNotes] = useState<string[]>([]);
  const [commentDraft, setCommentDraft] = useState<string>("");
  const [shareStatus, setShareStatus] = useState<"shared" | "copied" | "error" | null>(null);

  useEffect(() => {
    setOperatorNotes([]);
    setCommentDraft("");
    setShareStatus(null);
  }, [interceptId, detail?.report_markdown]);

  useEffect(() => {
    if (!shareStatus) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      setShareStatus(null);
    }, 2800);
    return () => window.clearTimeout(timeoutId);
  }, [shareStatus]);

  const baseReport = detail?.report_markdown ?? "";

  const augmentedReport = useMemo(() => {
    if (operatorNotes.length === 0) {
      return baseReport;
    }
    const heading = operatorNotes.length === 1 ? "## Operator Comment" : "## Operator Comments";
    const noteList = operatorNotes.map((note) => `- ${note}`).join("\n");
    const trimmedBase = baseReport.trimEnd();
    const prefix = trimmedBase.length > 0 ? `${trimmedBase}\n\n---\n\n` : "";
    return `${prefix}${heading}\n${noteList}`;
  }, [baseReport, operatorNotes]);

  const handleAppendNote = useCallback(() => {
    const trimmed = commentDraft.trim();
    if (!trimmed) {
      return;
    }
    const normalized = trimmed.replace(/\s*\n+\s*/g, " ");
    setOperatorNotes((prev) => [...prev, normalized]);
    setCommentDraft("");
  }, [commentDraft]);

  const handleShare = useCallback(async () => {
    if (!augmentedReport.trim()) {
      setShareStatus("error");
      return;
    }
    try {
      if ("share" in navigator && typeof navigator.share === "function") {
        await navigator.share({
          title: stripFileExtension(detail?.meta.title ?? "") || "Intercept Intelligence Report",
          text: augmentedReport,
        });
        setShareStatus("shared");
        return;
      }
      if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
        await navigator.clipboard.writeText(augmentedReport);
        setShareStatus("copied");
        return;
      }
      setShareStatus("error");
    } catch (err) {
      setShareStatus("error");
    }
  }, [augmentedReport, detail?.meta.title]);

  const appendDisabled = commentDraft.trim().length === 0;
  const shareDisabled = augmentedReport.trim().length === 0;

  if (!interceptId) {
    return (
      <div className="detail-deep detail-deep--empty">
        <span>Choose an intercept to view the timeline analysis.</span>
      </div>
    );
  }

  if (loading && !detail) {
    return (
      <div className="detail-deep detail-deep--loading">
        <span>Decrypting timeline intelligence…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="detail-deep detail-deep--error">
        <strong>Unable to load intercept detail.</strong>
        <span>{error}</span>
      </div>
    );
  }

  if (!detail) {
    return null;
  }

  const { meta } = detail;
  const timelineEmpty =
    detail.transcript.length === 0 && detail.background_events.length === 0;

  return (
    <div className="detail-deep">
      <div className="detail-deep__report-row">
        <section className="detail-deep__report">
          <header>
            <h3>Intelligence Report</h3>
          </header>
          <div className="detail-panel__report detail-panel__report--tall">
            <ReactMarkdown>{augmentedReport}</ReactMarkdown>
          </div>
        </section>

        <aside className="operator-console">
          <header className="operator-console__header">
            <h3>Operator Console</h3>
            <button
              type="button"
              className="operator-console__share"
              onClick={() => {
                void handleShare();
              }}
              disabled={shareDisabled}
            >
              Share Report
            </button>
          </header>
          <p className="operator-console__hint">
            Add mission context or clarifications. Notes append to the intelligence report for this
            intercept.
          </p>
          <textarea
            className="operator-console__textarea"
            placeholder="Document field impressions, tasking reminders, or follow-up actions…"
            value={commentDraft}
            onChange={(event) => setCommentDraft(event.target.value)}
          />
          <div className="operator-console__actions">
            <button
              type="button"
              className="operator-console__append"
              disabled={appendDisabled}
              onClick={handleAppendNote}
            >
              Append Comment
            </button>
            <span className="operator-console__counter">
              {operatorNotes.length === 0
                ? "No comments appended"
                : `${operatorNotes.length} comment${operatorNotes.length === 1 ? "" : "s"} appended`}
            </span>
          </div>
          {operatorNotes.length > 0 ? (
            <ul className="operator-console__notes">
              {operatorNotes.map((note, index) => (
                <li key={`${note}-${index}`}>{note}</li>
              ))}
            </ul>
          ) : (
            <span className="operator-console__placeholder">Notes appear here after append.</span>
          )}
          {shareStatus ? (
            <span className={`operator-console__status operator-console__status--${shareStatus}`}>
              {shareStatus === "shared"
                ? "Report shared via system dialog"
                : shareStatus === "copied"
                ? "Report copied to clipboard"
                : "Unable to share report"}
            </span>
          ) : null}
        </aside>
      </div>

      <section className="detail-deep__timeline">
        <header>
          <h3>Timeline Intelligence</h3>
          <span className="detail-deep__timeline-duration">
            {meta.duration_s ? formatSeconds(meta.duration_s) : "—"}
          </span>
        </header>
        <ThreatBulletin detail={detail} />
        {timelineEmpty ? (
          <div className="detail-panel__timeline-placeholder">
            {meta.status === "processing"
              ? "Timeline data is compiling..."
              : detail.report_markdown || "Timeline data unavailable."}
          </div>
        ) : (
          <MissionTimeline
            transcript={detail.transcript}
            events={detail.background_events}
            duration={meta.duration_s}
          />
        )}
      </section>
    </div>
  );
}

function computeAudioUrl(detail: InterceptDetail | null, interceptId: string | null) {
  if (!detail || !interceptId) {
    return null;
  }
  if (detail.meta.audio_url) {
    return detail.meta.audio_url;
  }
  return `/api/intercepts/${interceptId}/audio`;
}

function formatSeconds(duration: number): string {
  if (!Number.isFinite(duration)) return "-";
  const minutes = Math.floor(duration / 60);
  const seconds = Math.floor(duration % 60);
  return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
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

function ThreatBulletin({ detail }: { detail: InterceptDetail }) {
  const criticalTags = detail.meta.tags.filter((tag) => tag === "high-threat" || tag === "priority");
  const highEvents = detail.background_events
    .filter((event) => (event.probability ?? 0) >= 0.4)
    .slice(0, 4);

  if (criticalTags.length === 0 && highEvents.length === 0) {
    return null;
  }

  return (
    <div className="threat-bulletin">
      <span className="threat-bulletin__label">Threat Bulletin</span>
      {criticalTags.map((tag) => (
        <span key={tag} className="threat-bulletin__pill threat-bulletin__pill--alert">
          {tag}
        </span>
      ))}
      {highEvents.map((event, index) => (
        <span key={`${event.label}-${index}`} className="threat-bulletin__pill">
          {event.label} · {formatRange(event.start_s, event.end_s)} ·
          {event.probability ? ` ${Math.round(event.probability * 100)}%` : " n/a"}
        </span>
      ))}
    </div>
  );
}
