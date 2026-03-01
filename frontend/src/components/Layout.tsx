import { NavLink, Outlet } from 'react-router-dom';

const navItems = [
  { to: '/voice', label: 'Voice Setup' },
  { to: '/persona', label: 'Persona Config' },
  { to: '/meeting', label: 'Meeting Dashboard' },
  { to: '/settings', label: 'Settings' },
];

export default function Layout() {
  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>Zoom Auto</h1>
          <span className="version">v0.1.0</span>
        </div>
        <nav>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
