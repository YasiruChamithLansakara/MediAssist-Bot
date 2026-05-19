import { StrictMode, Component } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";

/* =========================================
   GLOBAL CSS IMPORT ORDER (CRITICAL)
   ========================================= */

/* 1. Design System */
import "./styles/variables.css";

/* 2. Base styles */
import "./styles/global.css";

/* 3. Layout system */
import "./styles/layout.css";

/* 4. Feature modules */
import "./styles/chat.css";
import "./styles/ocr.css";
import "./styles/drug.css";

/* =========================================
   ERROR BOUNDARY
   ========================================= */
class RootErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error, info) {
    console.error("UI Error:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: "20px", textAlign: "center" }}>
          <h2>⚠️ Something went wrong in the UI</h2>
          <p>Please refresh the page or check console logs.</p>
        </div>
      );
    }

    return this.props.children;
  }
}

/* =========================================
   APP RENDER
   ========================================= */
createRoot(document.getElementById("root")).render(
  <StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </StrictMode>,
);
