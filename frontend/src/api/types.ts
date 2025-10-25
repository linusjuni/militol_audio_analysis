export interface InterceptSummary {
  intercept_id: string;
  title: string;
  status: "processing" | "ready" | "failed" | "archived" | string;
  created_at: string;
  updated_at: string;
  duration_s?: number | null;
  priority?: string | null;
  source?: string | null;
  tags: string[];
  executive_summary?: string | null;
  audio_url?: string | null;
}

export interface TranscriptSegment {
  start_s: number;
  end_s: number;
  speaker: string;
  text: string;
}

export interface BackgroundEvent {
  label: string;
  start_s: number;
  end_s: number;
  probability?: number | null;
}

export interface InterceptDetail {
  meta: InterceptSummary;
  transcript: TranscriptSegment[];
  background_events: BackgroundEvent[];
  report_markdown: string;
}
