/**
 * useAppStore.ts
 * --------------
 * Small global UI-state store (Zustand). Server data itself (wells, curves,
 * crossplots) is fetched/cached with React Query in each component --
 * this store only holds cross-cutting UI state like "is the chat panel
 * open" and per-page chat conversation history.
 */

import { create } from "zustand";
import type { ChatMessage } from "@/api/types";

interface AppState {
  chatOpen: boolean;
  toggleChat: () => void;
  setChatOpen: (open: boolean) => void;

  // Conversation history keyed by scope ("dashboard" or a well_id), so the
  // field-wide chat and each well's chat keep independent histories.
  conversations: Record<string, ChatMessage[]>;
  appendMessage: (scope: string, message: ChatMessage) => void;
  clearConversation: (scope: string) => void;

  // The well/seismic-dataset currently "active" across pages -- set once
  // (e.g. by the Dashboard's combined upload) and read by the Seismic and
  // Synthetic Seismogram pages (and the chat panel's well_id context) so
  // a newly uploaded well/dataset appears everywhere without manual
  // re-selection on each page. Pages still allow a manual override; this
  // only seeds/redirects their local selection state.
  activeWellId: string | null;
  activeDatasetId: string | null;
  setActiveWell: (wellId: string | null, datasetId?: string | null) => void;
}

export const useAppStore = create<AppState>((set) => ({
  chatOpen: true,
  toggleChat: () => set((s) => ({ chatOpen: !s.chatOpen })),
  setChatOpen: (open) => set({ chatOpen: open }),

  conversations: {},
  appendMessage: (scope, message) =>
    set((s) => ({
      conversations: {
        ...s.conversations,
        [scope]: [...(s.conversations[scope] ?? []), message],
      },
    })),
  clearConversation: (scope) =>
    set((s) => ({ conversations: { ...s.conversations, [scope]: [] } })),

  activeWellId: null,
  activeDatasetId: null,
  setActiveWell: (wellId, datasetId) =>
    set((s) => ({
      activeWellId: wellId,
      activeDatasetId: datasetId !== undefined ? datasetId : s.activeDatasetId,
    })),
}));
