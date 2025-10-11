import React, { useState } from "react";
import { Download, Copy, CheckCircle, Book, FileCode, Terminal, Rocket, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Migration() {
  const [copiedSection, setCopiedSection] = useState(null);

  const copyToClipboard = (text, section) => {
    navigator.clipboard.writeText(text);
    setCopiedSection(section);
    setTimeout(() => setCopiedSection(null), 2000);
  };

  return (
    <div className="min-h-screen p-4 pb-20">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <div 
            className="w-10 h-10 rounded-xl flex items-center justify-center shadow-lg"
            style={{ 
              background: 'linear-gradient(to bottom right, rgb(var(--theme-primary)), rgb(var(--theme-secondary)))',
              boxShadow: '0 10px 25px rgba(var(--theme-primary), 0.25)'
            }}
          >
            <Download className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-white">Migration Guide</h1>
            <p className="text-sm text-white/60">Run VediGoldAI locally on your laptop (Already Downloaded as ZIP)</p>
          </div>
        </div>
      </div>

      {/* Quick Start Card */}
      <div 
        className="rounded-2xl p-6 backdrop-blur-xl mb-6"
        style={{ 
          background: 'linear-gradient(135deg, rgba(251, 191, 36, 0.1), rgba(249, 115, 22, 0.1))',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
          border: '1px solid rgba(251, 191, 36, 0.2)'
        }}
      >
        <div className="flex items-start gap-4">
          <Rocket className="w-8 h-8 text-amber-400 flex-shrink-0 mt-1" />
          <div>
            <h2 className="text-xl font-bold text-white mb-2">You've Downloaded the ZIP - Now What?</h2>
            <p className="text-white/70 text-sm mb-4">
              Follow these steps to set up and run your VediGoldAI app locally. 
              Total time: ~10 minutes
            </p>
            <div className="flex gap-2">
              <div className="px-3 py-1 rounded-lg bg-white/10 text-xs text-white">
                ⏱️ 10 minutes
              </div>
              <div className="px-3 py-1 rounded-lg bg-white/10 text-xs text-white">
                ⭐ Easy
              </div>
              <div className="px-3 py-1 rounded-lg bg-white/10 text-xs text-white">
                📦 Already Downloaded
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Prerequisites */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-4">
          <AlertCircle className="w-5 h-5 text-amber-400" />
          <h2 className="text-lg font-bold text-white">Prerequisites</h2>
        </div>
        <div 
          className="rounded-xl p-4 backdrop-blur-xl"
          style={{ 
            background: 'rgba(0, 0, 0, 0.3)',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)'
          }}
        >
          <p className="text-sm text-white/70 mb-3">Make sure you have these installed on your laptop:</p>
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-white">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span><strong>Node.js v18+</strong> - Download from <a href="https://nodejs.org/" target="_blank" rel="noopener noreferrer" className="text-amber-400 hover:underline">nodejs.org</a></span>
            </div>
            <div className="flex items-center gap-2 text-sm text-white">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span><strong>npm</strong> (comes with Node.js)</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-white">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span><strong>Code Editor</strong> (VS Code recommended)</span>
            </div>
          </div>
          <div className="mt-4 p-3 rounded-lg bg-black/40 border border-white/10">
            <p className="text-xs text-white/50 mb-2">Verify installations:</p>
            <code className="text-xs text-green-400">
              node --version<br/>
              npm --version
            </code>
          </div>
        </div>
      </div>

      {/* Step-by-Step Instructions */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-4">
          <Terminal className="w-5 h-5 text-amber-400" />
          <h2 className="text-lg font-bold text-white">Step-by-Step Setup</h2>
        </div>

        {/* Step 1: Extract ZIP */}
        <div 
          className="rounded-xl p-4 backdrop-blur-xl mb-4"
          style={{ 
            background: 'rgba(0, 0, 0, 0.3)',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)'
          }}
        >
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-xs">
              1
            </span>
            Extract the ZIP File
          </h3>
          <div className="space-y-2 ml-8">
            <p className="text-xs text-white/70">Extract the downloaded ZIP to a folder, for example:</p>
            <div className="p-3 rounded-lg bg-black/40 border border-white/10">
              <code className="text-xs text-green-400">
                C:\Users\YourName\Projects\vedigoldai<br/>
                or<br/>
                ~/Projects/vedigoldai
              </code>
            </div>
          </div>
        </div>

        {/* Step 2: Create React App */}
        <div 
          className="rounded-xl p-4 backdrop-blur-xl mb-4"
          style={{ 
            background: 'rgba(0, 0, 0, 0.3)',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)'
          }}
        >
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-xs">
              2
            </span>
            Create a New React + Vite Project
          </h3>
          <div className="space-y-3 ml-8">
            <p className="text-xs text-white/70">Open terminal and run:</p>
            <div className="space-y-2">
              <div className="flex items-center justify-between gap-3 p-3 rounded-lg bg-black/40 border border-white/10">
                <code className="text-xs text-green-400 flex-1">npm create vite@latest vedigoldai-local -- --template react</code>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => copyToClipboard('npm create vite@latest vedigoldai-local -- --template react', 'step2-1')}
                  className="h-7 w-7 p-0"
                >
                  {copiedSection === 'step2-1' ? (
                    <CheckCircle className="w-3 h-3 text-green-400" />
                  ) : (
                    <Copy className="w-3 h-3 text-white/50" />
                  )}
                </Button>
              </div>
              <div className="flex items-center justify-between gap-3 p-3 rounded-lg bg-black/40 border border-white/10">
                <code className="text-xs text-green-400 flex-1">cd vedigoldai-local</code>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => copyToClipboard('cd vedigoldai-local', 'step2-2')}
                  className="h-7 w-7 p-0"
                >
                  {copiedSection === 'step2-2' ? (
                    <CheckCircle className="w-3 h-3 text-green-400" />
                  ) : (
                    <Copy className="w-3 h-3 text-white/50" />
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Step 3: Copy Files */}
        <div 
          className="rounded-xl p-4 backdrop-blur-xl mb-4"
          style={{ 
            background: 'rgba(0, 0, 0, 0.3)',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)'
          }}
        >
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-xs">
              3
            </span>
            Copy Your Downloaded Files
          </h3>
          <div className="space-y-3 ml-8">
            <p className="text-xs text-white/70 mb-2">From your extracted ZIP, copy these folders INTO the new project's <code className="px-1 py-0.5 rounded bg-black/40 text-amber-400">src/</code> folder:</p>
            <div className="space-y-2">
              <div className="p-3 rounded-lg bg-black/40 border border-white/10">
                <code className="text-xs text-green-400">
                  ✅ Copy <strong>pages/</strong> folder → to <strong>src/pages/</strong><br/>
                  ✅ Copy <strong>components/</strong> folder → to <strong>src/components/</strong><br/>
                  ✅ Copy <strong>layout.js</strong> file → rename to <strong>src/layout/Layout.jsx</strong><br/>
                  ✅ Copy <strong>utils.js</strong> file → to <strong>src/utils/utils.js</strong>
                </code>
              </div>
            </div>
            <p className="text-xs text-white/50 mt-2">💡 Create the <code className="px-1 py-0.5 rounded bg-black/40">src/layout/</code> and <code className="px-1 py-0.5 rounded bg-black/40">src/utils/</code> folders if they don't exist.</p>
          </div>
        </div>

        {/* Step 4: Install Dependencies */}
        <div 
          className="rounded-xl p-4 backdrop-blur-xl mb-4"
          style={{ 
            background: 'rgba(0, 0, 0, 0.3)',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)'
          }}
        >
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-xs">
              4
            </span>
            Install Required Dependencies
          </h3>
          <div className="space-y-2 ml-8">
            <p className="text-xs text-white/70 mb-2">Run these commands in your terminal (inside the project folder):</p>
            <div className="space-y-2">
              {[
                'npm install react-router-dom @tanstack/react-query',
                'npm install lucide-react date-fns lodash recharts',
                'npm install class-variance-authority clsx tailwind-merge',
                'npm install @radix-ui/react-select @radix-ui/react-slot',
                'npm install -D tailwindcss postcss autoprefixer',
                'npx tailwindcss init -p'
              ].map((cmd, idx) => (
                <div key={idx} className="flex items-center justify-between gap-3 p-3 rounded-lg bg-black/40 border border-white/10">
                  <code className="text-xs text-green-400 flex-1">{cmd}</code>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => copyToClipboard(cmd, `step4-${idx}`)}
                    className="h-7 w-7 p-0"
                  >
                    {copiedSection === `step4-${idx}` ? (
                      <CheckCircle className="w-3 h-3 text-green-400" />
                    ) : (
                      <Copy className="w-3 h-3 text-white/50" />
                    )}
                  </Button>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Step 5: Code Changes */}
        <div 
          className="rounded-xl p-4 backdrop-blur-xl mb-4"
          style={{ 
            background: 'rgba(0, 0, 0, 0.3)',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)'
          }}
        >
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-xs">
              5
            </span>
            Make Required Code Changes
          </h3>
          <div className="space-y-3 ml-8">
            <p className="text-xs text-white/70 mb-2">Open these files in your code editor and make the following changes:</p>
            
            <div className="space-y-4">
              {/* Change 1: Layout imports */}
              <div className="p-3 rounded-lg bg-black/40 border border-white/10">
                <p className="text-xs font-semibold text-amber-400 mb-2">📝 src/layout/Layout.jsx</p>
                <div className="space-y-2">
                  <div>
                    <p className="text-xs text-red-400 mb-1">❌ Remove:</p>
                    <code className="text-xs text-red-400">import &#123; createPageUrl &#125; from "./utils"</code>
                  </div>
                  <div>
                    <p className="text-xs text-green-400 mb-1">✅ Replace with:</p>
                    <code className="text-xs text-green-400">import &#123; createPageUrl &#125; from "../utils/utils"</code>
                  </div>
                </div>
              </div>

              {/* Change 2: Remove base44 imports */}
              <div className="p-3 rounded-lg bg-black/40 border border-white/10">
                <p className="text-xs font-semibold text-amber-400 mb-2">📝 All component files</p>
                <div className="space-y-2">
                  <div>
                    <p className="text-xs text-red-400 mb-1">❌ Remove all lines with:</p>
                    <code className="text-xs text-red-400">
                      import &#123; base44 &#125; from '@/api/base44Client'<br/>
                      import &#123; base44 &#125; from "../api/base44Client"
                    </code>
                  </div>
                  <p className="text-xs text-white/50 mt-2">💡 Your API services already handle this with fetch() calls</p>
                </div>
              </div>

              {/* Change 3: Update imports to use @ alias */}
              <div className="p-3 rounded-lg bg-black/40 border border-white/10">
                <p className="text-xs font-semibold text-amber-400 mb-2">📝 Component imports</p>
                <div className="space-y-2">
                  <p className="text-xs text-white/70 mb-1">Change relative imports to use @ alias:</p>
                  <div className="mb-2">
                    <p className="text-xs text-red-400">❌ OLD:</p>
                    <code className="text-xs text-red-400">import Button from "../../components/ui/button"</code>
                  </div>
                  <div>
                    <p className="text-xs text-green-400">✅ NEW:</p>
                    <code className="text-xs text-green-400">import &#123; Button &#125; from "@/components/ui/button"</code>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Step 6: Create Required Files */}
        <div 
          className="rounded-xl p-4 backdrop-blur-xl mb-4"
          style={{ 
            background: 'rgba(0, 0, 0, 0.3)',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)'
          }}
        >
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-xs">
              6
            </span>
            Create Required Configuration Files
          </h3>
          <div className="space-y-3 ml-8">
            <p className="text-xs text-white/70 mb-2">Create these files in your project root (copy the code exactly):</p>
            
            {/* Show config files */}
            <div className="space-y-3">
              {[
                {
                  file: 'vite.config.js',
                  code: `import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    open: true,
  },
})`
                },
                {
                  file: 'tailwind.config.js',
                  code: `export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}`
                },
                {
                  file: '.env',
                  code: `VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCK_DATA=true`
                },
                {
                  file: 'src/main.jsx',
                  code: `import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient()

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
)`
                },
                {
                  file: 'src/App.jsx',
                  code: `import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './layout/Layout'
import LiveMonitor from './pages/LiveMonitor'
import Strategies from './pages/Strategies'
import SignalHistory from './pages/SignalHistory'
import Backtest from './pages/Backtest'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/live-monitor" replace />} />
        <Route path="/live-monitor" element={<Layout currentPageName="LiveMonitor"><LiveMonitor /></Layout>} />
        <Route path="/strategies" element={<Layout currentPageName="Strategies"><Strategies /></Layout>} />
        <Route path="/signal-history" element={<Layout currentPageName="SignalHistory"><SignalHistory /></Layout>} />
        <Route path="/backtest" element={<Layout currentPageName="Backtest"><Backtest /></Layout>} />
      </Routes>
    </BrowserRouter>
  )
}

export default App`
                },
                {
                  file: 'src/index.css',
                  code: `@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --theme-primary: 251, 191, 36;
  --theme-secondary: 249, 115, 22;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Inter', -apple-system, sans-serif;
  background-color: #000000;
  color: #ffffff;
}`
                }
              ].map((item, idx) => (
                <div key={idx} className="rounded-lg bg-black/60 border border-white/10 overflow-hidden">
                  <div className="p-2 bg-white/5 border-b border-white/10 flex items-center justify-between">
                    <code className="text-xs text-amber-400">{item.file}</code>
                    <Button
                      size="sm"
                      onClick={() => copyToClipboard(item.code, `config-${idx}`)}
                      className="h-6 text-xs bg-amber-500/20 hover:bg-amber-500/30 text-amber-400"
                    >
                      {copiedSection === `config-${idx}` ? (
                        <>
                          <CheckCircle className="w-3 h-3 mr-1" />
                          Copied!
                        </>
                      ) : (
                        <>
                          <Copy className="w-3 h-3 mr-1" />
                          Copy
                        </>
                      )}
                    </Button>
                  </div>
                  <div className="p-3 overflow-x-auto">
                    <pre className="text-xs text-green-400">
                      <code>{item.code}</code>
                    </pre>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Step 7: Run the App */}
        <div 
          className="rounded-xl p-4 backdrop-blur-xl mb-4"
          style={{ 
            background: 'rgba(0, 0, 0, 0.3)',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)'
          }}
        >
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-6 h-6 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-xs">
              7
            </span>
            Run Your App Locally
          </h3>
          <div className="space-y-3 ml-8">
            <p className="text-xs text-white/70 mb-2">Start the development server:</p>
            <div className="flex items-center justify-between gap-3 p-3 rounded-lg bg-black/40 border border-white/10">
              <code className="text-xs text-green-400 flex-1">npm run dev</code>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => copyToClipboard('npm run dev', 'step7')}
                className="h-7 w-7 p-0"
              >
                {copiedSection === 'step7' ? (
                  <CheckCircle className="w-3 h-3 text-green-400" />
                ) : (
                  <Copy className="w-3 h-3 text-white/50" />
                )}
              </Button>
            </div>
            <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/30">
              <p className="text-xs text-green-400">
                ✅ Your app will open automatically at <strong>http://localhost:3000</strong>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Important Notes */}
      <div 
        className="rounded-2xl p-6 backdrop-blur-xl"
        style={{ 
          background: 'rgba(59, 130, 246, 0.1)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
          border: '1px solid rgba(59, 130, 246, 0.2)'
        }}
      >
        <div className="flex items-start gap-4">
          <Book className="w-6 h-6 text-blue-400 flex-shrink-0" />
          <div>
            <h3 className="text-lg font-bold text-white mb-3">Important Notes</h3>
            <ul className="space-y-2 text-sm text-white/70">
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-1">•</span>
                <span>The app runs in <strong className="text-white">MOCK DATA mode</strong> by default. To connect to your backend, change <code className="px-2 py-0.5 rounded bg-black/40 text-green-400">VITE_USE_MOCK_DATA=false</code> in .env file.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-1">•</span>
                <span>Make sure your <strong className="text-white">FastAPI backend</strong> is running on <code className="px-2 py-0.5 rounded bg-black/40 text-green-400">http://localhost:8000</code> with CORS enabled.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-1">•</span>
                <span>All authentication and base44 platform features are removed - this is a standalone app.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-400 mt-1">•</span>
                <span>To build for production, run <code className="px-2 py-0.5 rounded bg-black/40 text-green-400">npm run build</code> - output goes to <code className="px-2 py-0.5 rounded bg-black/40 text-green-400">dist/</code> folder.</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Troubleshooting */}
      <div className="mt-6">
        <div className="flex items-center gap-2 mb-4">
          <AlertCircle className="w-5 h-5 text-red-400" />
          <h2 className="text-lg font-bold text-white">Common Issues & Solutions</h2>
        </div>
        <div className="space-y-3">
          {[
            {
              issue: "Module not found errors",
              solution: "Run 'npm install' again to ensure all dependencies are installed"
            },
            {
              issue: "Port 3000 already in use",
              solution: "Change port in vite.config.js or stop other apps using port 3000"
            },
            {
              issue: "Import errors with @/ alias",
              solution: "Make sure vite.config.js has the correct alias configuration"
            },
            {
              issue: "Tailwind styles not working",
              solution: "Ensure tailwind.config.js is in project root and index.css has @tailwind directives"
            },
            {
              issue: "API calls failing",
              solution: "Check that backend is running and VITE_API_BASE_URL in .env is correct"
            }
          ].map((item, idx) => (
            <div 
              key={idx}
              className="rounded-xl p-4 backdrop-blur-xl"
              style={{ 
                background: 'rgba(0, 0, 0, 0.3)',
                boxShadow: '0 4px 16px rgba(0, 0, 0, 0.1)'
              }}
            >
              <div className="flex items-start gap-3">
                <span className="text-red-400 text-sm mt-0.5">❌</span>
                <div className="flex-1">
                  <p className="text-sm font-semibold text-white mb-1">{item.issue}</p>
                  <p className="text-xs text-white/60"><strong className="text-green-400">Solution:</strong> {item.solution}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}