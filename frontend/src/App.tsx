import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { InterceptSummary } from "./api/types";
import { fetchIntercepts, uploadIntercept } from "./api/intercepts";
import { OpsShell } from "./components/OpsShell";
import { InterceptLogPanel } from "./components/InterceptLogPanel";
import { InterceptDetailPanel } from "./components/InterceptDetailPanel";
import type { QuickFilterKey } from "./constants/filters";
import { QUICK_FILTERS } from "./constants/filters";

function App() {
  const [intercepts, setIntercepts] = useState<InterceptSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [detailRefreshToken, setDetailRefreshToken] = useState<number>(0);
  const [activeFilters, setActiveFilters] = useState<QuickFilterKey[]>([]);
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

  const handleSelect = useCallback((id: string | null) => {
    selectedRef.current = id;
    setSelectedId((prev) => {
      if (prev === id) {
        return prev;
      }
      setDetailRefreshToken((token) => token + 1);
      return id;
    });
  }, []);

  const toggleFilter = useCallback((key: QuickFilterKey) => {
    setActiveFilters((prev) =>
      prev.includes(key) ? prev.filter((item) => item !== key) : [...prev, key],
    );
  }, []);

  const handleUpload = useCallback(
    async (file: File) => {
      try {
        setUploading(true);
        setUploadError(null);
        const summary = await uploadIntercept(file);
        setIntercepts((prev) => {
          const filtered = prev.filter((item) => item.intercept_id !== summary.intercept_id);
          const updated = [summary, ...filtered];
          interceptsRef.current = updated;
          return updated;
        });
        handleSelect(summary.intercept_id);
        window.setTimeout(() => {
          void loadIntercepts({ silent: true });
        }, 2500);
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : "Upload failed");
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
      uploadError={uploadError}
      activeFilters={activeFilters}
      onToggleFilter={toggleFilter}
      onUpload={handleUpload}
      filtersDescription={activeFilterMeta}
      detailPanel={
        <InterceptDetailPanel
          interceptId={selectedId}
          refreshToken={detailRefreshToken}
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
