//! Sliding-window conversation context. Mirrors `flow/agent/context.py`.
//!
//! The window bound is the memory budget: at most `max_messages` messages
//! are retained, oldest dropped first. Message content itself is not
//! truncated here — callers (the runtime) bound content upstream.

use std::collections::VecDeque;

/// Default window: 40 messages, like Python's `max_messages=40`.
pub const DEFAULT_MAX_MESSAGES: usize = 40;

/// One conversation message: a role (`"user"`, `"assistant"`, ...) and text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// Message role, e.g. `"user"`.
    pub role: String,
    /// Message text.
    pub content: String,
}

impl ChatMessage {
    /// Build a message; takes ownership of both parts.
    pub fn new(role: String, content: String) -> Self {
        Self { role, content }
    }
}

/// Bounded sliding-window message store.
#[derive(Debug, Clone)]
pub struct ConversationContext {
    max_messages: usize,
    messages: VecDeque<ChatMessage>,
}

impl ConversationContext {
    /// New context retaining at most `max_messages` messages. A zero cap
    /// discards everything, like `deque(maxlen=0)`.
    pub fn new(max_messages: usize) -> Self {
        Self {
            max_messages,
            messages: VecDeque::with_capacity(max_messages.min(1024)),
        }
    }

    /// New context with the default 40-message window.
    pub fn with_default_cap() -> Self {
        Self::new(DEFAULT_MAX_MESSAGES)
    }

    /// Append a message, evicting the oldest when the window is full.
    pub fn add_message(&mut self, role: &str, content: &str) {
        if self.max_messages == 0 {
            return;
        }
        while self.messages.len() >= self.max_messages {
            self.messages.pop_front();
        }
        self.messages
            .push_back(ChatMessage::new(role.to_owned(), content.to_owned()));
    }

    /// A snapshot of the retained messages, oldest first.
    pub fn messages(&self) -> Vec<ChatMessage> {
        self.messages.iter().cloned().collect()
    }

    /// Drop all retained messages.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Number of retained messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// True when no messages are retained.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// The configured window size.
    pub fn max_messages(&self) -> usize {
        self.max_messages
    }
}

impl Default for ConversationContext {
    fn default() -> Self {
        Self::with_default_cap()
    }
}
