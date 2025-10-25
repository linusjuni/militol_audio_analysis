import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import { fetchInterceptDetail } from "../api/intercepts";
import type { InterceptDetail } from "../api/types";
import { MissionTimeline } from "./MissionTimeline";
import "./InterceptDetailPanel.css";

interface InterceptDetailPanelProps {
  interceptId: string | null;
  refreshToken?: number;
}

export function InterceptDetailPanel({ interceptId, refreshToken }: InterceptDetailPanelProps) {
  const [detail, setDetail] = useState<InterceptDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!interceptId) {
      setDetail(null);
      setError(null);
      setLoading(false);
      return;
    }

    const load = async () => {
      try {
        setLoading(true);
        const payload = await fetchInterceptDetail(interceptId);
        setDetail(payload);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unable to load intercept detail");
      } finally {
        setLoading(false);
      }
    };

    void load();
  }, [interceptId, refreshToken]);

  const audioHref = useMemo(() => {
    if (!interceptId) {
      return null;
    }
    return `/api/intercepts/${interceptId}/audio`;
  }, [interceptId]);

  if (!interceptId) {
    return (
      <div className="detail-panel detail-panel--empty">
        <span>Select an intercept to open full analysis.</span>
      </div>
    );
  }

  if (loading && !detail) {
    return (
      <div className="detail-panel detail-panel--loading">
        <span>Decrypting and loading intercept…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="detail-panel detail-panel--error">
        <strong>Unable to load intercept.</strong>
        <span>{error}</span>
      </div>
    );
  }

  if (!detail) {
    return null;
  }

  const { meta } = detail;
  const startedAt = new Date(meta.created_at).toLocaleString();

  return (
    <div className="detail-panel">
      <header className="detail-panel__header">
        <div>
          <span className="detail-panel__eyebrow">ANALYSIS {meta.status.toUpperCase()}</span>
          <h2>{meta.title}</h2>
          <div className="detail-panel__meta">
            <span>Captured · {startedAt}</span>
            {meta.duration_s ? <span>Duration · {formatSeconds(meta.duration_s)}</span> : null}
          </div>
        </div>
        <div className="detail-panel__tags">
          {meta.tags.map((tag) => (
            <span key={tag}>{tag}</span>
          ))}
        </div>
      </header>

      {audioHref && (
        <div className="detail-panel__audio">
          <audio controls src={audioHref}>
            <track kind="captions" />
          </audio>
        </div>
      )}

      <section className="detail-panel__section">
        <h3>Timeline Intelligence</h3>
        {detail.transcript.length === 0 && detail.background_events.length === 0 ? (
          <div className="detail-panel__timeline-placeholder">
            Timeline data is compiling. Check back shortly.
          </div>
        ) : (
          <MissionTimeline
            transcript={detail.transcript}
            events={detail.background_events}
            duration={meta.duration_s}
          />
        )}
      </section>

      <section className="detail-panel__section">
        <h3>Intelligence Report</h3>
        <div className="detail-panel__report">
          <ReactMarkdown>{detail.report_markdown}</ReactMarkdown>
        </div>
      </section>
    </div>
  );
}

function formatSeconds(duration: number): string {
  if (!Number.isFinite(duration)) return "-";
  const minutes = Math.floor(duration / 60);
  const seconds = Math.floor(duration % 60);
  return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
}
