import type { ReactNode } from "react";
import type { InterceptSummary } from "../api/types";
import type { QuickFilterKey } from "../constants/filters";
import { QUICK_FILTERS } from "../constants/filters";
import "./OpsShell.css";

interface OpsShellProps {
  intercepts: InterceptSummary[];
  loading: boolean;
  refreshing: boolean;
  error: string | null;
  uploading: boolean;
  opsError: string | null;
  consoleMessage: string | null;
  activeFilters: QuickFilterKey[];
  onToggleFilter: (filter: QuickFilterKey) => void;
  filtersDescription: string[];
  onUpload: (file: File) => void | Promise<void>;
  overviewPanel: ReactNode;
  detailPanel: ReactNode;
  children: ReactNode;
}

export function OpsShell({
  intercepts,
  children,
  loading,
  refreshing,
  error,
  uploading,
  opsError,
  consoleMessage,
  activeFilters,
  onToggleFilter,
  filtersDescription,
  onUpload,
  overviewPanel,
  detailPanel,
}: OpsShellProps) {
  const readyCount = intercepts.filter((it) => it.status === "ready").length;
  const processingCount = intercepts.filter((it) => it.status === "processing").length;
  const failedCount = intercepts.filter((it) => it.status === "failed").length;

  return (
    <div className="ops-shell">
      <div className="ops-shell__top">
        <aside className="ops-shell__left">
          <div className="ops-shell__brand">
            <div className="ops-shell__badge">MILITOL</div>
            <div className="ops-shell__brand-text">
              <span>Signals Command</span>
              <small>Audio Intelligence Console</small>
            </div>
          </div>

          <div className="ops-shell__pulse">
            <PulseItem label="Ready" value={readyCount} />
            <PulseItem label="Processing" value={processingCount} />
            <PulseItem label="Issues" value={failedCount} tone="danger" />
          </div>

          <div className="ops-shell__actions">
            <button
              className="ops-shell__action-button"
              type="button"
              onClick={() => {
                const uploadInput = document.getElementById("ops-shell-upload");
                uploadInput?.click();
              }}
            >
              Queue New Intercept
            </button>
            <input
              id="ops-shell-upload"
              type="file"
              accept=".wav,audio/*"
              hidden
              onChange={(event) => {
                const [file] = event.target.files ?? [];
                if (file) {
                  void onUpload(file);
                  event.target.value = "";
                }
              }}
            />
          </div>

          <div className="ops-shell__status">
            {loading && <span className="ops-shell__status-loading">Syncing intercepts…</span>}
            {uploading && <span className="ops-shell__status-loading">Processing upload…</span>}
            {opsError && <span className="ops-shell__status-error">{opsError}</span>}
            {error && <span className="ops-shell__status-error">{error}</span>}
            {filtersDescription.length > 0 && (
              <span className="ops-shell__status-filter">
                Filters active: {filtersDescription.join(", ")}
              </span>
            )}
            {!loading && !error && (
              <div className="ops-shell__status-row">
                <span className="ops-shell__status-idle">
                  {consoleMessage && !refreshing
                    ? consoleMessage
                    : `${intercepts.length} intercept${intercepts.length === 1 ? "" : "s"} in log`}
                </span>
                {refreshing && (
                  <span className="ops-shell__status-refresh" role="status" aria-live="polite">
                    <span className="ops-shell__status-spinner" aria-hidden="true" />
                    Refreshing
                  </span>
                )}
              </div>
            )}
          </div>

          <div className="ops-shell__filters">
            <h4>Quick Filters</h4>
            <div className="ops-shell__filter-chips">
              {QUICK_FILTERS.map((filter) => {
                const active = activeFilters.includes(filter.key);
                return (
                  <button
                    key={filter.key}
                    type="button"
                    className={`ops-shell__filter-button ${
                      active ? "ops-shell__filter-button--active" : ""
                    }`}
                    onClick={() => onToggleFilter(filter.key)}
                  >
                    {filter.label}
                  </button>
                );
              })}
            </div>
          </div>
        </aside>

        <main className="ops-shell__log-area">{children}</main>

        <section className="ops-shell__overview-area">{overviewPanel}</section>
      </div>

      <section className="ops-shell__detail-area">{detailPanel}</section>
    </div>
  );
}

function PulseItem({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone?: "danger";
}) {
  return (
    <div className="ops-shell__pulse-item">
      <span className="ops-shell__pulse-label">{label}</span>
      <span
        className="ops-shell__pulse-value"
        style={tone === "danger" ? { color: "var(--color-danger)" } : undefined}
      >
        {value}
      </span>
    </div>
  );
}
