import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { InterceptDetail, InterceptSummary } from "./api/types";
import {
  deleteIntercept,
  fetchInterceptDetail,
  fetchIntercepts,
  rerunIntercept,
  uploadIntercept,
} from "./api/intercepts";
import { OpsShell } from "./components/OpsShell";
import { InterceptLogPanel } from "./components/InterceptLogPanel";
import {
  InterceptDeepDivePanel,
  InterceptOverviewPanel,
} from "./components/InterceptDetailPanel";
import type { QuickFilterKey } from "./constants/filters";
import { QUICK_FILTERS } from "./constants/filters";

function App() {
  const [intercepts, setIntercepts] = useState<InterceptSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [opsError, setOpsError] = useState<string | null>(null);
  const [consoleMessage, setConsoleMessage] = useState<string | null>(null);
  const [detailRefreshToken, setDetailRefreshToken] = useState<number>(0);
  const [activeFilters, setActiveFilters] = useState<QuickFilterKey[]>([]);
  const [detail, setDetail] = useState<InterceptDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState<boolean>(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [actionBusyId, setActionBusyId] = useState<string | null>(null);
  const interceptsRef = useRef<InterceptSummary[]>([]);
  const selectedRef = useRef<string | null>(null);

  const loadIntercepts = useCallback(async (opts?: { silent?: boolean }) => {
    try {
      if (opts?.silent) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }
      const data = await fetchIntercepts();
      const previousIntercepts = interceptsRef.current;
      setIntercepts(data);

      const currentSelected = selectedRef.current;
      const nextSelectedId =
        currentSelected && data.some((item) => item.intercept_id === currentSelected)
          ? currentSelected
          : data.length > 0
          ? data[0].intercept_id
          : null;

      if (nextSelectedId !== currentSelected) {
        setSelectedId(nextSelectedId);
      }
      selectedRef.current = nextSelectedId;

      if (nextSelectedId) {
        const previousSelectedMeta = previousIntercepts.find(
          (item) => item.intercept_id === nextSelectedId,
        );
        const refreshedMeta = data.find((item) => item.intercept_id === nextSelectedId);
        if (
          previousSelectedMeta &&
          refreshedMeta &&
          previousSelectedMeta.status !== refreshedMeta.status
        ) {
          setDetailRefreshToken((token) => token + 1);
        }
      }
      interceptsRef.current = data;
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load intercepts");
    } finally {
      if (opts?.silent) {
        setRefreshing(false);
      } else {
        setLoading(false);
      }
    }
  }, []);

  const filteredIntercepts = useMemo(
    () => applyFilters(intercepts, activeFilters),
    [intercepts, activeFilters],
  );

  useEffect(() => {
    if (activeFilters.length === 0) {
      return;
    }
    if (filteredIntercepts.length === 0) {
      if (selectedRef.current !== null) {
        selectedRef.current = null;
        setSelectedId(null);
      }
      return;
    }
    const current = selectedRef.current;
    if (!current || !filteredIntercepts.some((item) => item.intercept_id === current)) {
      const fallback = filteredIntercepts[0].intercept_id;
      selectedRef.current = fallback;
      setSelectedId((prev) => {
        if (prev === fallback) {
          return prev;
        }
        setDetailRefreshToken((token) => token + 1);
        return fallback;
      });
    }
  }, [filteredIntercepts, activeFilters]);

  useEffect(() => {
    void loadIntercepts();
  }, [loadIntercepts]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      void loadIntercepts({ silent: true });
    }, 10000);
    return () => window.clearInterval(interval);
  }, [loadIntercepts]);

  useEffect(() => {
    selectedRef.current = selectedId;
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      setDetailError(null);
      setDetailLoading(false);
      return;
    }

    let cancelled = false;
    const load = async () => {
      try {
        setDetailLoading(true);
        const payload = await fetchInterceptDetail(selectedId);
        if (!cancelled) {
          setDetail(payload);
          setDetailError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setDetailError(err instanceof Error ? err.message : "Unable to load intercept detail");
        }
      } finally {
        if (!cancelled) {
          setDetailLoading(false);
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [selectedId, detailRefreshToken]);

  const handleSelect = useCallback((id: string | null) => {
    selectedRef.current = id;
    setSelectedId((prev) => {
      if (prev === id) {
        return prev;
      }
      setDetail(null);
      setDetailError(null);
      setDetailRefreshToken((token) => token + 1);
      return id;
    });
  }, []);

  const toggleFilter = useCallback((key: QuickFilterKey) => {
    setActiveFilters((prev) =>
      prev.includes(key) ? prev.filter((item) => item !== key) : [...prev, key],
    );
  }, []);

  const handleDelete = useCallback(
    async (interceptId: string) => {
      try {
        setActionBusyId(interceptId);
        setOpsError(null);
        setConsoleMessage("Removing intercept...");
        await deleteIntercept(interceptId);

        setIntercepts((prev) => {
          const updated = prev.filter((item) => item.intercept_id !== interceptId);
          interceptsRef.current = updated;
          if (selectedRef.current === interceptId) {
            const fallback = updated[0]?.intercept_id ?? null;
            selectedRef.current = fallback;
            setSelectedId(fallback);
            setDetail(null);
          }
          return updated;
        });

        void loadIntercepts({ silent: true });
        setConsoleMessage("Intercept removed.");
        window.setTimeout(() => setConsoleMessage(null), 2500);
      } catch (err) {
        setOpsError(err instanceof Error ? err.message : "Failed to remove intercept");
        setConsoleMessage(null);
      } finally {
        setActionBusyId((current) => (current === interceptId ? null : current));
      }
    },
    [loadIntercepts],
  );

  const handleRerun = useCallback(
    async (interceptId: string) => {
      try {
        setActionBusyId(interceptId);
        setOpsError(null);
        setConsoleMessage("Re-running pipeline...");
        const summary = await rerunIntercept(interceptId);
        setIntercepts((prev) => {
          const updated = prev.map((item) =>
            item.intercept_id === summary.intercept_id ? summary : item,
          );
          interceptsRef.current = updated;
          return updated;
        });

        if (selectedRef.current === interceptId) {
          setDetail((prev) =>
            prev
              ? {
                  ...prev,
                  meta: { ...prev.meta, ...summary },
                  transcript: [],
                  background_events: [],
                  report_markdown: "Analysis in progress...",
                }
              : prev,
          );
          setDetailLoading(true);
          setDetailError(null);
          setDetailRefreshToken((token) => token + 1);
        }

        void loadIntercepts({ silent: true });
        setConsoleMessage("Pipeline restarted.");
        window.setTimeout(() => setConsoleMessage(null), 2500);
      } catch (err) {
        setOpsError(err instanceof Error ? err.message : "Failed to re-run pipeline");
        setConsoleMessage(null);
      } finally {
        setActionBusyId((current) => (current === interceptId ? null : current));
      }
    },
    [loadIntercepts],
  );

  const handleUpload = useCallback(
    async (file: File) => {
      try {
        setUploading(true);
        setOpsError(null);
        setConsoleMessage("Uploading intercept...");
        const summary = await uploadIntercept(file);
        setIntercepts((prev) => {
          const filtered = prev.filter((item) => item.intercept_id !== summary.intercept_id);
          const updated = [summary, ...filtered];
          interceptsRef.current = updated;
          return updated;
        });
        handleSelect(summary.intercept_id);
        setConsoleMessage("Intercept queued for analysis.");
        window.setTimeout(() => setConsoleMessage(null), 2500);
        window.setTimeout(() => {
          void loadIntercepts({ silent: true });
        }, 2500);
      } catch (err) {
        setOpsError(err instanceof Error ? err.message : "Upload failed");
        setConsoleMessage(null);
      } finally {
        setUploading(false);
      }
    },
    [handleSelect, loadIntercepts],
  );

  const activeFilterMeta = useMemo(
    () =>
      QUICK_FILTERS.filter((filter) => activeFilters.includes(filter.key)).map(
        (filter) => filter.label,
      ),
    [activeFilters],
  );

  return (
    <OpsShell
      intercepts={intercepts}
      loading={loading}
      refreshing={refreshing}
      error={error}
      uploading={uploading}
      opsError={opsError}
      consoleMessage={consoleMessage}
      activeFilters={activeFilters}
      onToggleFilter={toggleFilter}
      onUpload={handleUpload}
      filtersDescription={activeFilterMeta}
      overviewPanel={
        <InterceptOverviewPanel
          detail={detail}
          loading={detailLoading}
          error={detailError}
          interceptId={selectedId}
          onDelete={handleDelete}
          onRerun={handleRerun}
          busyId={actionBusyId}
        />
      }
      detailPanel={
        <InterceptDeepDivePanel
          detail={detail}
          loading={detailLoading}
          error={detailError}
          interceptId={selectedId}
        />
      }
    >
      <InterceptLogPanel
        items={filteredIntercepts}
        selectedId={selectedId}
        onSelect={(id) => handleSelect(id)}
        loading={loading}
        totalCount={intercepts.length}
        filtersActive={activeFilters.length > 0}
      />
    </OpsShell>
  );
}

export default App;

function applyFilters(
  intercepts: InterceptSummary[],
  activeFilters: QuickFilterKey[],
): InterceptSummary[] {
  if (activeFilters.length === 0) {
    return intercepts;
  }

  const now = Date.now();
  return intercepts.filter((item) => {
    return activeFilters.every((filter) => {
      switch (filter) {
        case "flagged":
          return (
            item.status === "failed" ||
            (item.priority ?? "").toLowerCase() === "high" ||
            item.tags.includes("high-threat") ||
            item.tags.includes("priority")
          );
        case "recent": {
          const created = Date.parse(item.created_at);
          if (Number.isNaN(created)) {
            return false;
          }
          const diffHours = (now - created) / (1000 * 60 * 60);
          return diffHours <= 24;
        }
        case "background": {
          if (item.tags.includes("background-alert")) {
            return true;
          }
          const summary = (item.executive_summary ?? "").toLowerCase();
          return ["gunshot", "shell", "vehicle", "explosion"].some((keyword) =>
            summary.includes(keyword),
          );
        }
        default:
          return true;
      }
    });
  });
}
