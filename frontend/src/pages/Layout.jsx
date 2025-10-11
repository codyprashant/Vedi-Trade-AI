
import React from "react";
import { Link } from "react-router-dom";
import { Monitor, BarChart3, History, TestTube, Download, Menu, X } from "lucide-react";
import { createPageUrl } from "../utils";

export default function Layout({ children, currentPageName }) {
  const [sidebarOpen, setSidebarOpen] = React.useState(false);

  const navItems = [
    { name: "LiveMonitor", label: "Live Monitor", icon: Monitor, gradient: "from-red-500 to-rose-600" },
    { name: "Strategies", label: "Strategies", icon: BarChart3, gradient: "from-blue-500 to-cyan-600" },
    { name: "SignalHistory", label: "Signal History", icon: History, gradient: "from-purple-500 to-pink-600" },
    { name: "Backtest", label: "Backtest", icon: TestTube, gradient: "from-green-500 to-emerald-600" },
    { name: "Migration", label: "Migration Guide", icon: Download, gradient: "from-amber-500 to-orange-600" },
  ];

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: '#000000' }}>
      <style>{`
        :root {
          --theme-primary: 251, 191, 36;
          --theme-secondary: 249, 115, 22;
          --theme-accent: 245, 158, 11;
        }
        
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }
        
        body {
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
            'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

        @keyframes shimmer {
          0% { background-position: -1000px 0; }
          100% { background-position: 1000px 0; }
        }

        @keyframes pulse-glow {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .animate-shimmer {
          animation: shimmer 3s infinite;
          background: linear-gradient(
            to right,
            rgba(255, 255, 255, 0) 0%,
            rgba(255, 255, 255, 0.1) 50%,
            rgba(255, 255, 255, 0) 100%
          );
          background-size: 1000px 100%;
        }
      `}</style>

      {/* Mobile Menu Button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 w-10 h-10 rounded-xl flex items-center justify-center backdrop-blur-xl"
        style={{ 
          background: 'rgba(0, 0, 0, 0.5)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
        }}
      >
        {sidebarOpen ? (
          <X className="w-5 h-5 text-white" />
        ) : (
          <Menu className="w-5 h-5 text-white" />
        )}
      </button>

      {/* Sidebar */}
      <aside
        className={`fixed lg:relative inset-y-0 left-0 z-40 w-64 transform transition-transform duration-300 ease-in-out ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } lg:translate-x-0`}
        style={{
          background: 'linear-gradient(180deg, rgba(0, 0, 0, 0.95) 0%, rgba(10, 10, 10, 0.98) 100%)',
          boxShadow: '4px 0 24px rgba(0, 0, 0, 0.3)',
          borderRight: '1px solid rgba(255, 255, 255, 0.05)'
        }}
      >
        <div className="h-full flex flex-col p-4">
          {/* Logo */}
          <div className="mb-6 px-2">
            <div className="flex items-center gap-3">
              <div 
                className="w-10 h-10 rounded-xl flex items-center justify-center shadow-2xl"
                style={{ 
                  background: 'linear-gradient(135deg, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))',
                  boxShadow: '0 10px 30px rgba(var(--theme-primary), 0.3)'
                }}
              >
                <span className="text-xl font-bold text-white">V</span>
              </div>
              <div>
                <h1 className="text-lg font-bold text-white tracking-tight">VediGoldAI</h1>
                <p className="text-[10px] text-white/40 uppercase tracking-wider">Trading Bot</p>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = currentPageName === item.name;
              
              return (
                <Link
                  key={item.name}
                  to={createPageUrl(item.name)}
                  onClick={() => setSidebarOpen(false)}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-300 group relative overflow-hidden ${
                    isActive ? 'shadow-lg' : ''
                  }`}
                  style={{
                    background: isActive 
                      ? 'linear-gradient(135deg, rgba(var(--theme-primary), 0.15), rgba(var(--theme-secondary), 0.15))'
                      : 'transparent',
                    border: isActive 
                      ? '1px solid rgba(var(--theme-primary), 0.3)' 
                      : '1px solid transparent'
                  }}
                >
                  {isActive && (
                    <div 
                      className="absolute inset-0 animate-shimmer"
                      style={{
                        background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent)',
                        backgroundSize: '200% 100%',
                      }}
                    />
                  )}
                  
                  <div 
                    className={`w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-300 ${
                      isActive ? 'shadow-lg' : 'bg-white/5 group-hover:bg-white/10'
                    }`}
                    style={isActive ? {
                      background: `linear-gradient(135deg, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))`,
                      boxShadow: '0 4px 12px rgba(var(--theme-primary), 0.4)'
                    } : {}}
                  >
                    <Icon className={`w-4 h-4 ${isActive ? 'text-white' : 'text-white/60 group-hover:text-white/80'}`} />
                  </div>
                  
                  <span className={`text-sm font-medium transition-colors ${
                    isActive ? 'text-white' : 'text-white/60 group-hover:text-white/90'
                  }`}>
                    {item.label}
                  </span>
                  
                  {isActive && (
                    <div 
                      className="absolute right-2 w-1.5 h-1.5 rounded-full"
                      style={{
                        background: 'rgb(var(--theme-primary))',
                        boxShadow: '0 0 8px rgba(var(--theme-primary), 0.8)',
                        animation: 'pulse-glow 2s ease-in-out infinite'
                      }}
                    />
                  )}
                </Link>
              );
            })}
          </nav>

          {/* Footer */}
          <div 
            className="mt-auto pt-4 px-3 py-3 rounded-xl"
            style={{
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.05)'
            }}
          >
            <div className="text-[10px] text-white/40 text-center">
              <div className="font-semibold mb-1">VediGoldAI v1.0</div>
              <div>Powered by AI & Technical Analysis</div>
            </div>
          </div>
        </div>
      </aside>

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/60 backdrop-blur-sm z-30"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  );
}
