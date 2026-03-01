import { useCallback, useEffect, useRef, useState } from 'react';
import { meetingsApi } from '../api';
import type { DashboardState, DashboardWSMessage, MeetingStatusResponse } from '../types';

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function MeetingDashboard() {
  const [meetingId, setMeetingId] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('AI Assistant');
  const [status, setStatus] = useState<MeetingStatusResponse | null>(null);
  const [dashState, setDashState] = useState<DashboardState | null>(null);
  const [message, setMessage] = useState('');
  const [joining, setJoining] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // Load initial status
  const loadStatus = useCallback(async () => {
    try {
      const s = await meetingsApi.getStatus();
      setStatus(s);
    } catch {
      // Ignore errors on status polling
    }
  }, []);

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 5000);
    return () => clearInterval(interval);
  }, [loadStatus]);

  // WebSocket connection
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/dashboard/ws`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const msg: DashboardWSMessage = JSON.parse(event.data);
        if (msg.type === 'state' || msg.type === 'state_update') {
          if (msg.data) {
            setDashState(msg.data);
          }
        }
      } catch {
        // Ignore parse errors
      }
    };

    ws.onerror = () => {
      // WebSocket errors are expected when the backend is not running
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, []);

  // Auto-scroll transcript
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [dashState?.transcript]);

  const handleJoin = async () => {
    if (!meetingId.trim()) {
      setMessage('Meeting ID is required.');
      return;
    }
    setJoining(true);
    setMessage('');
    try {
      const res = await meetingsApi.join({
        meeting_id: meetingId.trim(),
        password,
        display_name: displayName,
      });
      setMessage(res.message);
      await loadStatus();
    } catch (err) {
      setMessage(`Join failed: ${err}`);
    } finally {
      setJoining(false);
    }
  };

  const handleLeave = async () => {
    setLeaving(true);
    setMessage('');
    try {
      const res = await meetingsApi.leave();
      setMessage(res.message);
      await loadStatus();
    } catch (err) {
      setMessage(`Leave failed: ${err}`);
    } finally {
      setLeaving(false);
    }
  };

  const connected = status?.connected || dashState?.connected || false;

  return (
    <div className="page">
      <h2>Meeting Dashboard</h2>

      {message && <div className="message-bar">{message}</div>}

      {/* Status Bar */}
      <div className={`status-banner ${connected ? 'status-ready' : 'status-pending'}`}>
        <strong>{connected ? 'Connected' : 'Disconnected'}</strong>
        {connected && status && (
          <span>
            {' '}&mdash; Meeting {status.meeting_id} | {status.participants} participants |{' '}
            {formatDuration(status.duration_seconds)} | {status.utterances_count} utterances |{' '}
            {status.responses_count} responses
          </span>
        )}
      </div>

      {/* Join Form (when disconnected) */}
      {!connected && (
        <section className="card">
          <h3>Join Meeting</h3>
          <div className="form-group">
            <label htmlFor="meeting-id">Meeting ID</label>
            <input
              id="meeting-id"
              type="text"
              placeholder="123 456 7890"
              value={meetingId}
              onChange={(e) => setMeetingId(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label htmlFor="meeting-password">Password</label>
            <input
              id="meeting-password"
              type="password"
              placeholder="Optional"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label htmlFor="display-name">Display Name</label>
            <input
              id="display-name"
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
            />
          </div>
          <button className="btn btn-primary" onClick={handleJoin} disabled={joining}>
            {joining ? 'Joining...' : 'Join Meeting'}
          </button>
        </section>
      )}

      {/* Meeting Controls (when connected) */}
      {connected && (
        <div className="action-bar">
          <button className="btn btn-danger" onClick={handleLeave} disabled={leaving}>
            {leaving ? 'Leaving...' : 'Leave Meeting'}
          </button>
        </div>
      )}

      {/* Live Transcript */}
      <section className="card">
        <h3>Live Transcript</h3>
        <div className="transcript-feed">
          {(!dashState || dashState.transcript.length === 0) ? (
            <p className="help-text">No transcript data yet. Join a meeting to see live updates.</p>
          ) : (
            dashState.transcript.map((entry, i) => (
              <div key={i} className="transcript-entry">
                <span className="transcript-speaker">{entry.speaker}</span>
                <span className="transcript-time">
                  {entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : ''}
                </span>
                <p className="transcript-text">{entry.text}</p>
              </div>
            ))
          )}
          <div ref={transcriptEndRef} />
        </div>
      </section>

      {/* Meeting State */}
      {dashState && (dashState.participants.length > 0 || dashState.decisions.length > 0) && (
        <div className="grid-2">
          {dashState.participants.length > 0 && (
            <section className="card">
              <h3>Participants ({dashState.participants.length})</h3>
              <ul className="simple-list">
                {dashState.participants.map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            </section>
          )}

          {dashState.decisions.length > 0 && (
            <section className="card">
              <h3>Decisions</h3>
              <ul className="simple-list">
                {dashState.decisions.map((d, i) => (
                  <li key={i}>{d}</li>
                ))}
              </ul>
            </section>
          )}
        </div>
      )}

      {dashState && dashState.action_items.length > 0 && (
        <section className="card">
          <h3>Action Items</h3>
          <ul className="simple-list">
            {dashState.action_items.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
