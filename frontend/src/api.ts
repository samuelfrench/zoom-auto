/** API client using native fetch. */

const BASE = '';

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`${resp.status}: ${err}`);
  }
  return resp.json() as Promise<T>;
}

// --- Voice ---
import type {
  PromptsResponse,
  SamplesListResponse,
  StatusResponse,
  CombineResponse,
  UploadResponse,
  PersonaConfig,
  PersonaUpdateRequest,
  MeetingJoinRequest,
  MeetingStatusResponse,
  DashboardState,
  AppSettings,
} from './types';

export const voiceApi = {
  getPrompts: () => request<PromptsResponse>('/api/voice/prompts'),
  getSamples: (user: string) => request<SamplesListResponse>(`/api/voice/samples/${user}`),
  getStatus: (user: string) => request<StatusResponse>(`/api/voice/status/${user}`),
  uploadSample: async (file: File, user: string, promptIndex: number, promptText: string): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    const params = new URLSearchParams({
      user,
      prompt_index: String(promptIndex),
      prompt_text: promptText,
    });
    const resp = await fetch(`/api/voice/upload?${params}`, {
      method: 'POST',
      body: formData,
    });
    if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
    return resp.json();
  },
  deleteSample: (user: string, segmentId: string) =>
    request(`/api/voice/samples/${user}/${segmentId}`, { method: 'DELETE' }),
  combineSamples: (user: string) =>
    request<CombineResponse>(`/api/voice/combine/${user}`, { method: 'POST' }),
};

// --- Persona ---
export const personaApi = {
  getConfig: () => request<PersonaConfig>('/api/persona/config'),
  updateConfig: (data: PersonaUpdateRequest) =>
    request<PersonaConfig>('/api/persona/config', {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  rebuild: () =>
    request<{ status: string; message: string }>('/api/persona/rebuild', { method: 'POST' }),
};

// --- Meetings ---
export const meetingsApi = {
  join: (data: MeetingJoinRequest) =>
    request<{ status: string; message: string }>('/api/meetings/join', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  leave: () =>
    request<{ status: string; message: string }>('/api/meetings/leave', { method: 'POST' }),
  getStatus: () => request<MeetingStatusResponse>('/api/meetings/status'),
};

// --- Dashboard ---
export const dashboardApi = {
  getState: () => request<DashboardState>('/api/dashboard/state'),
};

// --- Settings ---
export const settingsApi = {
  get: () => request<AppSettings>('/api/settings'),
};
