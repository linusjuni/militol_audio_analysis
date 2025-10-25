export type QuickFilterKey = "flagged" | "recent" | "background";

export const QUICK_FILTERS: Array<{ key: QuickFilterKey; label: string }> = [
  { key: "flagged", label: "Flagged Priority" },
  { key: "recent", label: "Last 24 Hours" },
  { key: "background", label: "Background Alerts" },
];
