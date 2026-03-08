import Link from 'next/link';
import { Activity, Settings, History } from 'lucide-react';

export default function Navbar() {
  return (
    <nav className="glass-panel" style={{ margin: '1rem', padding: '1rem 2rem', borderBottom: '1px solid var(--panel-border)', borderRadius: '16px' }}>
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Activity size={28} color="var(--accent-primary)" />
          <h1 style={{ fontSize: '1.5rem', margin: 0 }} className="text-gradient">Neurolens</h1>
        </div>
        
        <div className="flex gap-4">
          <Link href="/" className="btn btn-secondary">
            <Activity size={18} /> Dashboard
          </Link>
          <Link href="/history" className="btn btn-secondary">
            <History size={18} /> History
          </Link>
          <Link href="/settings" className="btn btn-secondary">
            <Settings size={18} /> Settings
          </Link>
        </div>
      </div>
    </nav>
  );
}
