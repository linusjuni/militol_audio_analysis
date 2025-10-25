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
  uploadError: string | null;
  activeFilters: QuickFilterKey[];
  onToggleFilter: (filter: QuickFilterKey) => void;
  filtersDescription: string[];
  onUpload: (file: File) => void | Promise<void>;
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
  uploadError,
  activeFilters,
  onToggleFilter,
  filtersDescription,
  onUpload,
  detailPanel,
}: OpsShellProps) {
  const readyCount = intercepts.filter((it) => it.status === "ready").length;
  const processingCount = intercepts.filter((it) => it.status === "processing").length;
  const failedCount = intercepts.filter((it) => it.status === "failed").length;

  return (
    <div className="ops-shell">
      <aside className="ops-shell__left">
        <div className="ops-shell__brand">
          <div className="ops-shell__badge">MILITOL</div>
          <div className="ops-shell__brand-text">
            <span>Signals Command</span>
            <small>Audio Intelligence Console</small>
          </div>
        </div>

        <div className="ops-shell__stat-grid">
          <StatChip label="Ready" value={readyCount} tone="positive" />
          <StatChip label="Processing" value={processingCount} tone="warning" />
          <StatChip label="Issues" value={failedCount} tone="danger" />
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
          {refreshing && !loading && (
            <span className="ops-shell__status-loading">Refreshing intercepts…</span>
          )}
          {uploading && <span className="ops-shell__status-loading">Processing upload…</span>}
          {uploadError && <span className="ops-shell__status-error">{uploadError}</span>}
          {error && <span className="ops-shell__status-error">{error}</span>}
          {filtersDescription.length > 0 && (
            <span className="ops-shell__status-filter">
              Filters active: {filtersDescription.join(", ")}
            </span>
          )}
          {!loading && !error && (
            <span className="ops-shell__status-idle">
              {intercepts.length} intercept{intercepts.length === 1 ? "" : "s"} in log
            </span>
          )}
        </div>

        <div className="ops-shell__filters">
          <h4>Quick Filters</h4>
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
      </aside>

      <main className="ops-shell__log-area">{children}</main>

      <section className="ops-shell__detail-area">{detailPanel}</section>
    </div>
  );
}

function StatChip({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone: "positive" | "warning" | "danger";
}) {
  return (
    <div className={`stat-chip stat-chip--${tone}`}>
      <span className="stat-chip__value">{value}</span>
      <span className="stat-chip__label">{label}</span>
    </div>
  );
}
