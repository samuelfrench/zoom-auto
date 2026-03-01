import { useCallback, useEffect, useState } from 'react';
import { personaApi } from '../api';
import type { PersonaConfig as PersonaConfigType } from '../types';

const TRAIT_FIELDS: { key: keyof PersonaConfigType; label: string; type: 'slider' }[] = [
  { key: 'formality', label: 'Formality', type: 'slider' },
  { key: 'verbosity', label: 'Verbosity', type: 'slider' },
  { key: 'technical_depth', label: 'Technical Depth', type: 'slider' },
  { key: 'assertiveness', label: 'Assertiveness', type: 'slider' },
  { key: 'vocabulary_richness', label: 'Vocabulary Richness', type: 'slider' },
  { key: 'question_frequency', label: 'Question Frequency', type: 'slider' },
  { key: 'exclamation_rate', label: 'Exclamation Rate', type: 'slider' },
];

export default function PersonaConfigPage() {
  const [config, setConfig] = useState<PersonaConfigType | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [message, setMessage] = useState('');
  const [dirty, setDirty] = useState(false);

  const loadConfig = useCallback(async () => {
    try {
      const data = await personaApi.getConfig();
      setConfig(data);
    } catch (err) {
      setMessage(`Failed to load persona: ${err}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  const handleChange = (key: string, value: string | number) => {
    if (!config) return;
    setConfig({ ...config, [key]: value });
    setDirty(true);
  };

  const handleSave = async () => {
    if (!config) return;
    setSaving(true);
    setMessage('');
    try {
      const updated = await personaApi.updateConfig(config);
      setConfig(updated);
      setDirty(false);
      setMessage('Persona saved successfully.');
    } catch (err) {
      setMessage(`Save failed: ${err}`);
    } finally {
      setSaving(false);
    }
  };

  const handleRebuild = async () => {
    setRebuilding(true);
    setMessage('');
    try {
      const res = await personaApi.rebuild();
      setMessage(res.message);
      await loadConfig();
      setDirty(false);
    } catch (err) {
      setMessage(`Rebuild failed: ${err}`);
    } finally {
      setRebuilding(false);
    }
  };

  if (loading) return <div className="page-loading">Loading persona...</div>;
  if (!config) return <div className="page-loading">No persona config available.</div>;

  return (
    <div className="page">
      <h2>Persona Configuration</h2>

      {message && <div className="message-bar">{message}</div>}

      {/* Identity */}
      <section className="card">
        <h3>Identity</h3>
        <div className="form-group">
          <label htmlFor="persona-name">Name</label>
          <input
            id="persona-name"
            type="text"
            value={config.name}
            onChange={(e) => handleChange('name', e.target.value)}
          />
        </div>
      </section>

      {/* Trait Sliders */}
      <section className="card">
        <h3>Communication Traits</h3>
        {TRAIT_FIELDS.map(({ key, label }) => (
          <div key={key} className="form-group slider-group">
            <label htmlFor={`trait-${key}`}>
              {label}: {(config[key] as number).toFixed(2)}
            </label>
            <input
              id={`trait-${key}`}
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={config[key] as number}
              onChange={(e) => handleChange(key, parseFloat(e.target.value))}
            />
          </div>
        ))}
      </section>

      {/* Response Style */}
      <section className="card">
        <h3>Response Style</h3>
        <div className="form-group">
          <label htmlFor="avg-words">Avg Response Words</label>
          <input
            id="avg-words"
            type="number"
            min="10"
            max="500"
            value={config.avg_response_words}
            onChange={(e) => handleChange('avg_response_words', parseInt(e.target.value) || 50)}
          />
        </div>
        <div className="form-group">
          <label htmlFor="greeting">Greeting Style</label>
          <input
            id="greeting"
            type="text"
            value={config.greeting_style}
            onChange={(e) => handleChange('greeting_style', e.target.value)}
          />
        </div>
        <div className="form-group">
          <label htmlFor="agreement">Agreement Style</label>
          <input
            id="agreement"
            type="text"
            value={config.agreement_style}
            onChange={(e) => handleChange('agreement_style', e.target.value)}
          />
        </div>
        <div className="form-group">
          <label htmlFor="standup">Standup Format</label>
          <textarea
            id="standup"
            rows={3}
            value={config.standup_format}
            onChange={(e) => handleChange('standup_format', e.target.value)}
          />
        </div>
      </section>

      {/* Terms */}
      <section className="card">
        <h3>Vocabulary</h3>
        <div className="form-group">
          <label>Preferred Terms</label>
          <div className="tag-list">
            {config.preferred_terms.map((t, i) => (
              <span key={i} className="tag">{t}</span>
            ))}
            {config.preferred_terms.length === 0 && <span className="help-text">None</span>}
          </div>
        </div>
        <div className="form-group">
          <label>Common Phrases</label>
          <div className="tag-list">
            {config.common_phrases.map((p, i) => (
              <span key={i} className="tag">{p}</span>
            ))}
            {config.common_phrases.length === 0 && <span className="help-text">None</span>}
          </div>
        </div>
        <div className="form-group">
          <label>Filler Words</label>
          <div className="tag-list">
            {Object.entries(config.filler_words).map(([word, rate]) => (
              <span key={word} className="tag">{word}: {rate.toFixed(1)}/100w</span>
            ))}
            {Object.keys(config.filler_words).length === 0 && <span className="help-text">None</span>}
          </div>
        </div>
      </section>

      {/* Actions */}
      <div className="action-bar">
        <button className="btn btn-primary" onClick={handleSave} disabled={saving || !dirty}>
          {saving ? 'Saving...' : 'Save Changes'}
        </button>
        <button className="btn btn-secondary" onClick={handleRebuild} disabled={rebuilding}>
          {rebuilding ? 'Rebuilding...' : 'Rebuild from Sources'}
        </button>
      </div>
    </div>
  );
}
