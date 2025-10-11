import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient()

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <QueryClientProvider data-filename="pages/Migration" data-linenumber="400" data-visual-selector-id="pages/Migration400" client={queryClient}>
      <App data-filename="pages/Migration" data-linenumber="401" data-visual-selector-id="pages/Migration401" />
    </QueryClientProvider>
  </React.StrictMode>,
)