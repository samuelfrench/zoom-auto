import { useCallback, useEffect, useRef, useState } from 'react';
import { voiceApi } from '../api';
import type { PromptItem, SegmentResponse, StatusResponse } from '../types';

const USER = 'default';

export default function VoiceSetup() {
  const [prompts, setPrompts] = useState<PromptItem[]>([]);
  const [segments, setSegments] = useState<SegmentResponse[]>([]);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [combining, setCombining] = useState(false);
  const [message, setMessage] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadData = useCallback(async () => {
    try {
      const [promptsRes, samplesRes, statusRes] = await Promise.all([
        voiceApi.getPrompts(),
        voiceApi.getSamples(USER),
        voiceApi.getStatus(USER),
      ]);
      setPrompts(promptsRes.prompts);
      setSegments(samplesRes.segments);
      setStatus(statusRes);
    } catch (err) {
      setMessage(`Failed to load data: ${err}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setUploading(true);
    setMessage('');
    try {
      for (const file of Array.from(files)) {
        const res = await voiceApi.uploadSample(file, USER, -1, '');
        setMessage(res.message);
      }
      await loadData();
    } catch (err) {
      setMessage(`Upload failed: ${err}`);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (segmentId: string) => {
    try {
      await voiceApi.deleteSample(USER, segmentId);
      await loadData();
    } catch (err) {
      setMessage(`Delete failed: ${err}`);
    }
  };

  const handleCombine = async () => {
    setCombining(true);
    setMessage('');
    try {
      const res = await voiceApi.combineSamples(USER);
      setMessage(res.message);
      await loadData();
    } catch (err) {
      setMessage(`Combine failed: ${err}`);
    } finally {
      setCombining(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleUpload(e.dataTransfer.files);
  };

  if (loading) return <div className="page-loading">Loading voice data...</div>;

  return (
    <div className="page">
      <h2>Voice Setup</h2>

      {/* Status Banner */}
      {status && (
        <div className={`status-banner ${status.ready_for_tts ? 'status-ready' : 'status-pending'}`}>
          <strong>{status.ready_for_tts ? 'Ready for TTS' : 'Not Ready'}</strong>
          <span> &mdash; {status.valid_segments}/{status.total_segments} valid segments, {status.total_valid_duration_seconds.toFixed(1)}s total</span>
          {status.recommendation && <p className="recommendation">{status.recommendation}</p>}
        </div>
      )}

      {message && <div className="message-bar">{message}</div>}

      {/* Recording Prompts */}
      <section className="card">
        <h3>Recording Prompts</h3>
        <p className="help-text">Read these prompts aloud and upload the recordings below.</p>
        <div className="prompts-list">
          {prompts.map((p) => (
            <div key={p.index} className="prompt-item">
              <span className="prompt-index">#{p.index + 1}</span>
              <span className="prompt-text">{p.text}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Upload */}
      <section className="card">
        <h3>Upload Audio</h3>
        <div
          className={`drop-zone ${dragOver ? 'drop-zone-active' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <p>{uploading ? 'Uploading...' : 'Drop audio files here or click to browse'}</p>
          <p className="help-text">Supports WAV, MP3, FLAC, M4A (max 50MB)</p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.flac,.m4a"
            multiple
            style={{ display: 'none' }}
            onChange={(e) => handleUpload(e.target.files)}
          />
        </div>
      </section>

      {/* Segments List */}
      {segments.length > 0 && (
        <section className="card">
          <h3>Uploaded Segments ({segments.length})</h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>File</th>
                <th>Duration</th>
                <th>SNR</th>
                <th>Valid</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {segments.map((seg) => (
                <tr key={seg.segment_id}>
                  <td>{seg.filename}</td>
                  <td>{seg.duration_seconds.toFixed(1)}s</td>
                  <td>{seg.snr_db.toFixed(1)} dB</td>
                  <td className={seg.is_valid ? 'valid' : 'invalid'}>
                    {seg.is_valid ? 'Yes' : 'No'}
                  </td>
                  <td>
                    <button className="btn-small btn-danger" onClick={() => handleDelete(seg.segment_id)}>
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <button
            className="btn btn-primary"
            onClick={handleCombine}
            disabled={combining || segments.filter(s => s.is_valid).length === 0}
          >
            {combining ? 'Combining...' : 'Combine into Reference WAV'}
          </button>
        </section>
      )}
    </div>
  );
}
