import { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

/**
 * Catches rendering errors in any page/component beneath it and shows a
 * friendly, light-themed fallback instead of letting React unmount the
 * whole tree (which otherwise shows as a blank/crashed page with no
 * indication of what went wrong). Wraps the routed page content in App.tsx.
 */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // eslint-disable-next-line no-console
    console.error("Unhandled error in page render:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="max-w-xl mx-auto mt-16 bg-surface border border-red-200 rounded-xl p-6 text-center shadow-card">
          <p className="text-sm font-semibold text-danger mb-1">Something went wrong</p>
          <p className="text-xs text-ink-muted mb-4">{this.state.error.message}</p>
          <button
            onClick={() => this.setState({ error: null })}
            className="text-xs font-semibold px-3.5 py-1.5 rounded-full bg-accent text-white hover:bg-accent-strong transition-colors"
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
