import type { InterceptDetail, InterceptSummary } from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "/api";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchIntercepts(): Promise<InterceptSummary[]> {
  const response = await fetch(`${API_BASE}/intercepts`);
  return parseJson<InterceptSummary[]>(response);
}

export async function fetchInterceptDetail(interceptId: string): Promise<InterceptDetail> {
  const response = await fetch(`${API_BASE}/intercepts/${interceptId}`);
  return parseJson<InterceptDetail>(response);
}

export async function uploadIntercept(file: File): Promise<InterceptSummary> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE}/intercepts`, {
    method: "POST",
    body: formData,
  });
  return parseJson<InterceptSummary>(response);
}

export async function deleteIntercept(interceptId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/intercepts/${interceptId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Failed to delete intercept (${response.status})`);
  }
}

export async function rerunIntercept(interceptId: string): Promise<InterceptSummary> {
  const response = await fetch(`${API_BASE}/intercepts/${interceptId}/rerun`, {
    method: "POST",
  });
  return parseJson<InterceptSummary>(response);
}
