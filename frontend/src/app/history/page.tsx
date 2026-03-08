import { History } from "lucide-react";

export default function HistoryPage() {
  return (
    <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <div className="flex items-center gap-2" style={{ marginBottom: '2rem' }}>
        <History size={32} color="var(--accent-primary)" />
        <h2 style={{ margin: 0 }}>Session History</h2>
      </div>

      <div className="glass-panel" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
        <History size={64} style={{ opacity: 0.2, margin: '0 auto 1rem' }} />
        <h3 style={{ color: 'var(--text-secondary)' }}>Session Recording Required</h3>
        <p style={{ color: 'var(--text-muted)' }}>
          To view detailed past sessions, the metrics need to be persisted to a database. 
          Currently, Neurolens operates in real-time memory mode. 
          Use the Live Dashboard to view analytics for your active session.
        </p>
      </div>
    </div>
  );
}
