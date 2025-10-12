
export const APP_CONFIG = {
  // Toggle between mock data and live server
  USE_MOCK_DATA: false, // Disabled to use live backend
  
  // Backend API base URL
  API_BASE_URL: 'http://localhost:8000',
  
  // Signal limits for different pages
  SIGNAL_LIMITS: {
    LIVE_MONITOR: 10,
    DASHBOARD: 5,
    SIGNAL_HISTORY: 20
  }
};
