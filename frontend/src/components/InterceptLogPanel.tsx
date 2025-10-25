import type { InterceptSummary } from "../api/types";
import "./InterceptLogPanel.css";

interface InterceptLogPanelProps {
  items: InterceptSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  loading: boolean;
  totalCount: number;
  filtersActive: boolean;
}

export function InterceptLogPanel({
  items,
  selectedId,
  onSelect,
  loading,
  totalCount,
  filtersActive,
}: InterceptLogPanelProps) {
  if (loading && items.length === 0) {
    return <div className="log-panel__empty">Loading intercept log…</div>;
  }

  if (!loading && items.length === 0) {
    return (
      <div className="log-panel__empty">
        {filtersActive
          ? "No intercepts match the current filters."
          : "No intercepts yet. Queue a new file to begin analysis."}
      </div>
    );
  }

  return (
    <div className="log-panel">
      <div className="log-panel__header">
        <div>
          <h2>Intercept Log</h2>
          <span className="log-panel__subtitle">Mission feed · auto-refreshing</span>
        </div>
        <div className="log-panel__chips">
          <span className="log-panel__chip">
            {items.length}
            {filtersActive ? ` of ${totalCount}` : " total"}
          </span>
        </div>
      </div>

      <div className="log-panel__list">
        {items.map((item) => (
          <article
            key={item.intercept_id}
            onClick={() => onSelect(item.intercept_id)}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onSelect(item.intercept_id);
              }
            }}
            className={`log-card ${selectedId === item.intercept_id ? "log-card--selected" : ""}`}
          >
            <header className="log-card__header">
              <div>
                <span className="log-card__title">{item.title}</span>
                <span className={`log-card__status log-card__status--${item.status}`}>
                  {item.status.toUpperCase()}
                </span>
              </div>
              <time dateTime={item.created_at}>
                {new Date(item.created_at).toLocaleString()}
              </time>
            </header>
            {item.executive_summary && (
              <p className="log-card__summary">{item.executive_summary}</p>
            )}
            {item.tags.length > 0 && (
              <div className="log-card__tags">
                {item.tags.map((tag) => (
                  <span key={tag} className="log-card__tag">
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </article>
        ))}
      </div>
    </div>
  );
}
