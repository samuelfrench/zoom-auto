import { useEffect, useState } from 'react';
import { settingsApi } from '../api';
import type { AppSettings } from '../types';

interface SettingsSectionProps {
  title: string;
  items: { label: string; value: string | number }[];
}

function SettingsSection({ title, items }: SettingsSectionProps) {
  return (
    <section className="card">
      <h3>{title}</h3>
      <dl className="settings-list">
        {items.map(({ label, value }) => (
          <div key={label} className="settings-item">
            <dt>{label}</dt>
            <dd><code>{String(value)}</code></dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    settingsApi.get()
      .then(setSettings)
      .catch((err) => setError(`Failed to load settings: ${err}`))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page-loading">Loading settings...</div>;
  if (error) return <div className="page"><div className="message-bar">{error}</div></div>;
  if (!settings) return null;

  return (
    <div className="page">
      <h2>Settings</h2>
      <p className="help-text">Current application configuration (read-only). Edit config/local.toml to change settings.</p>

      <SettingsSection
        title="Zoom"
        items={[
          { label: 'Bot Name', value: settings.zoom.bot_name },
        ]}
      />

      <SettingsSection
        title="LLM Provider"
        items={[
          { label: 'Provider', value: settings.llm.provider },
          { label: 'Response Model', value: settings.llm.response_model },
          { label: 'Decision Model', value: settings.llm.decision_model },
          { label: 'Max Tokens', value: settings.llm.max_tokens },
          { label: 'Temperature', value: settings.llm.temperature },
        ]}
      />

      <SettingsSection
        title="Speech-to-Text"
        items={[
          { label: 'Model', value: settings.stt.model },
          { label: 'Language', value: settings.stt.language },
          { label: 'Beam Size', value: settings.stt.beam_size },
        ]}
      />

      <SettingsSection
        title="Text-to-Speech"
        items={[
          { label: 'Voice Sample Dir', value: settings.tts.voice_sample_dir },
          { label: 'Sample Rate', value: `${settings.tts.sample_rate} Hz` },
        ]}
      />

      <SettingsSection
        title="Context Management"
        items={[
          { label: 'Max Window Tokens', value: settings.context.max_window_tokens },
          { label: 'Verbatim Window', value: `${settings.context.verbatim_window_seconds}s` },
          { label: 'Summary Interval', value: `${settings.context.summary_interval_seconds}s` },
        ]}
      />

      <SettingsSection
        title="Response Engine"
        items={[
          { label: 'Cooldown', value: `${settings.response.cooldown_seconds}s` },
          { label: 'Trigger Threshold', value: settings.response.trigger_threshold },
          { label: 'Max Consecutive', value: settings.response.max_consecutive },
        ]}
      />

      <SettingsSection
        title="Voice Activity Detection"
        items={[
          { label: 'Threshold', value: settings.vad.threshold },
          { label: 'Min Speech Duration', value: `${settings.vad.min_speech_duration}s` },
        ]}
      />
    </div>
  );
}
