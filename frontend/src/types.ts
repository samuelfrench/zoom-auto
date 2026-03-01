/** Shared TypeScript types for the Zoom Auto dashboard. */

// --- Voice ---

export interface PromptItem {
  index: number;
  text: string;
}

export interface PromptsResponse {
  prompts: PromptItem[];
  total: number;
}

export interface SegmentResponse {
  segment_id: string;
  filename: string;
  prompt_index: number;
  prompt_text: string;
  duration_seconds: number;
  snr_db: number;
  peak_amplitude: number;
  rms_amplitude: number;
  has_clipping: boolean;
  is_valid: boolean;
  recorded_at: string;
}

export interface QualityResponse {
  is_valid: boolean;
  snr_db: number;
  peak_amplitude: number;
  rms_amplitude: number;
  duration_seconds: number;
  has_clipping: boolean;
  issues: string[];
}

export interface UploadResponse {
  success: boolean;
  segment: SegmentResponse;
  quality: QualityResponse;
  message: string;
}

export interface SamplesListResponse {
  user: string;
  segments: SegmentResponse[];
  total: number;
}

export interface StatusResponse {
  user: string;
  total_segments: number;
  valid_segments: number;
  total_valid_duration_seconds: number;
  average_snr_db: number;
  has_combined_reference: boolean;
  combined_duration_seconds: number;
  ready_for_tts: boolean;
  recommendation: string;
}

export interface CombineResponse {
  success: boolean;
  combined_path: string;
  duration_seconds: number;
  message: string;
}

// --- Persona ---

export interface PersonaConfig {
  name: string;
  formality: number;
  verbosity: number;
  technical_depth: number;
  assertiveness: number;
  avg_response_words: number;
  greeting_style: string;
  agreement_style: string;
  filler_words: Record<string, number>;
  common_phrases: string[];
  preferred_terms: string[];
  avoided_terms: string[];
  standup_format: string;
  vocabulary_richness: number;
  question_frequency: number;
  exclamation_rate: number;
}

export interface PersonaUpdateRequest {
  name?: string;
  formality?: number;
  verbosity?: number;
  technical_depth?: number;
  assertiveness?: number;
  avg_response_words?: number;
  greeting_style?: string;
  agreement_style?: string;
  filler_words?: Record<string, number>;
  common_phrases?: string[];
  preferred_terms?: string[];
  avoided_terms?: string[];
  standup_format?: string;
  vocabulary_richness?: number;
  question_frequency?: number;
  exclamation_rate?: number;
}

// --- Meetings ---

export interface MeetingJoinRequest {
  meeting_id: string;
  password: string;
  display_name: string;
}

export interface MeetingStatusResponse {
  connected: boolean;
  meeting_id: string | null;
  participants: number;
  duration_seconds: number;
  utterances_count: number;
  responses_count: number;
}

// --- Dashboard ---

export interface TranscriptEntry {
  speaker: string;
  text: string;
  timestamp: string;
}

export interface DashboardState {
  connected: boolean;
  meeting_id: string | null;
  participants: string[];
  duration_seconds: number;
  transcript: TranscriptEntry[];
  decisions: string[];
  action_items: string[];
  bot_responses: Record<string, unknown>[];
}

export interface DashboardWSMessage {
  type: 'state' | 'state_update' | 'pong';
  data?: DashboardState;
  timestamp?: number;
}

// --- Settings ---

export interface AppSettings {
  zoom: {
    bot_name: string;
  };
  llm: {
    provider: string;
    response_model: string;
    decision_model: string;
    max_tokens: number;
    temperature: number;
  };
  stt: {
    model: string;
    language: string;
    beam_size: number;
  };
  tts: {
    voice_sample_dir: string;
    sample_rate: number;
  };
  context: {
    max_window_tokens: number;
    verbatim_window_seconds: number;
    summary_interval_seconds: number;
  };
  response: {
    cooldown_seconds: number;
    trigger_threshold: number;
    max_consecutive: number;
  };
  vad: {
    threshold: number;
    min_speech_duration: number;
  };
}
