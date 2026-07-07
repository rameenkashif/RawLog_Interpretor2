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
}));
