"""
MEMORY LAYER: Persistent AI Memory via Backboard.io

Provides context management for:
1. Session Memory - temporary context during single analysis pipeline
2. Caller Memory - persistent memory tied to caller identifier
3. Pattern Memory - aggregate patterns learned across all calls

Uses Backboard.io's memory="Auto" for intelligent context management.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from backboard import BackboardClient

BACKBOARD_API_KEY = os.getenv("BACKBOARD_API_KEY", "")


@dataclass
class SessionContext:
    """Holds context for a single analysis session."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    caller_id: Optional[str] = None
    thread_ids: dict = field(default_factory=dict)  # assistant_key -> thread_id
    context_data: dict = field(default_factory=dict)
    
    def is_expired(self, max_age_hours: int = 24) -> bool:
        return datetime.now() - self.created_at > timedelta(hours=max_age_hours)


@dataclass
class CallerMemory:
    """Persistent memory for a specific caller."""
    caller_id: str
    call_history: list = field(default_factory=list)
    patterns: dict = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def add_call(self, call_data: dict):
        self.call_history.append({
            **call_data,
            "recorded_at": datetime.now().isoformat()
        })
        # Keep last 50 calls
        if len(self.call_history) > 50:
            self.call_history = self.call_history[-50:]
        self.last_updated = datetime.now()
    
    def get_summary(self) -> str:
        """Generate a summary of caller history for context injection."""
        if not self.call_history:
            return "No previous call history."
        
        lines = [f"Previous interactions ({len(self.call_history)} calls):"]
        for i, call in enumerate(self.call_history[-5:], 1):  # Last 5 calls
            intent = call.get("intent", "unknown")
            risk = call.get("risk_level", "unknown")
            score = call.get("compliance_score")
            sentiment = call.get("sentiment", "")
            violations = call.get("violations_count", 0)
            timestamp = call.get("recorded_at", "unknown")[:10]
            
            # Build info line
            info_parts = [f"Intent: {intent}", f"Risk: {risk}"]
            if score is not None:
                info_parts.append(f"Score: {score}")
            if sentiment:
                info_parts.append(f"Sentiment: {sentiment}")
            if violations > 0:
                info_parts.append(f"Violations: {violations}")
            
            lines.append(f"  {i}. [{timestamp}] {', '.join(info_parts)}")
        
        return "\n".join(lines)


class MemoryManager:
    """
    Manages persistent AI memory via Backboard.io.
    
    Provides:
    - Session-based memory (temporary, for single pipeline run)
    - Caller-based memory (persistent across calls)
    - Pattern memory (aggregate learning)
    - Context retrieval for assistants
    """
    
    def __init__(self):
        self._sessions: dict[str, SessionContext] = {}
        self._caller_memories: dict[str, CallerMemory] = {}
        self._pattern_memory: dict[str, list] = {}
        self._client: Optional[BackboardClient] = None
        self._memory_assistant_id: Optional[str] = None
        self._initialized = False
    
    async def _get_client(self) -> BackboardClient:
        """Get or create Backboard client."""
        if self._client is None:
            key = os.getenv("BACKBOARD_API_KEY", BACKBOARD_API_KEY)
            if not key:
                raise RuntimeError("BACKBOARD_API_KEY not set")
            self._client = BackboardClient(api_key=key)
        return self._client
    
    async def initialize(self):
        """Initialize memory system with Backboard."""
        if self._initialized:
            return
        
        client = await self._get_client()
        
        # Create a memory-focused assistant for context summarization
        try:
            memory_assistant = await client.create_assistant(
                name="Memory Context Manager",
                system_prompt="""You are a memory management assistant that helps summarize and retrieve relevant context.

When given conversation history or call data, extract and organize:
1. Key facts about the caller (preferences, issues, history)
2. Important patterns (repeated complaints, compliance concerns)
3. Relevant context for the current interaction

Always respond with structured JSON containing:
- "key_facts": list of important facts
- "patterns": list of recurring patterns  
- "context_summary": brief text summary
- "relevance_score": 0-1 indicating how relevant this history is"""
            )
            self._memory_assistant_id = memory_assistant.assistant_id
        except Exception as e:
            print(f"Warning: Could not create memory assistant: {e}")
        
        self._initialized = True
    
    # ── Session Management ──────────────────────────────────────────────────
    
    async def create_session(self, session_id: str, caller_id: str = None) -> SessionContext:
        """Create a new analysis session."""
        session = SessionContext(
            session_id=session_id,
            caller_id=caller_id
        )
        self._sessions[session_id] = session
        
        # Clean up expired sessions
        self._cleanup_expired_sessions()
        
        return session
    
    async def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get an existing session."""
        session = self._sessions.get(session_id)
        if session and not session.is_expired():
            return session
        return None
    
    async def get_session_context(self, session_id: str) -> dict:
        """Get all context data for a session."""
        session = await self.get_session(session_id)
        if not session:
            return {}
        return session.context_data
    
    async def add_to_session(self, session_id: str, key: str, value: any):
        """Add data to session context."""
        session = await self.get_session(session_id)
        if session:
            session.context_data[key] = value
    
    async def get_or_create_thread(
        self,
        assistant_id: str,
        session_id: str = None,
        caller_id: str = None,
        use_persistent: bool = True
    ) -> str:
        """Get or create a thread for an assistant, optionally with persistence."""
        client = await self._get_client()
        
        if not use_persistent or not session_id:
            # Create fresh thread (current behavior)
            thread = await client.create_thread(assistant_id)
            return thread.thread_id
        
        session = await self.get_session(session_id)
        if session and assistant_id in session.thread_ids:
            return session.thread_ids[assistant_id]
        
        # Create new thread and store in session
        thread = await client.create_thread(assistant_id)
        
        if session:
            session.thread_ids[assistant_id] = thread.thread_id
        
        return thread.thread_id
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        expired = [
            sid for sid, session in self._sessions.items() 
            if session.is_expired()
        ]
        for sid in expired:
            del self._sessions[sid]
    
    # ── Caller Memory ───────────────────────────────────────────────────────
    
    async def get_caller_memory(self, caller_id: str) -> Optional[CallerMemory]:
        """Get memory for a specific caller."""
        return self._caller_memories.get(caller_id)
    
    async def get_caller_history(self, caller_id: str) -> list[dict]:
        """Get call history for a caller."""
        memory = await self.get_caller_memory(caller_id)
        if not memory:
            return []
        return memory.call_history
    
    async def store_call_summary(self, caller_id: str, call_data: dict):
        """Store a call summary in caller memory."""
        if not caller_id:
            return
        
        if caller_id not in self._caller_memories:
            self._caller_memories[caller_id] = CallerMemory(caller_id=caller_id)
        
        self._caller_memories[caller_id].add_call(call_data)
    
    async def get_caller_context(self, caller_id: str) -> str:
        """Get formatted context string from caller history."""
        if not caller_id:
            return ""
        
        memory = await self.get_caller_memory(caller_id)
        if not memory:
            return ""
        
        return memory.get_summary()
    
    async def clear_caller_memory(self, caller_id: str) -> bool:
        """Clear all memory for a caller (GDPR compliance)."""
        if caller_id in self._caller_memories:
            del self._caller_memories[caller_id]
            return True
        return False
    
    # ── Pattern Memory ──────────────────────────────────────────────────────
    
    async def store_pattern(self, pattern_type: str, pattern: dict):
        """Store a detected pattern for aggregate learning."""
        if pattern_type not in self._pattern_memory:
            self._pattern_memory[pattern_type] = []
        
        self._pattern_memory[pattern_type].append({
            **pattern,
            "stored_at": datetime.now().isoformat()
        })
        
        # Keep last 100 patterns per type
        if len(self._pattern_memory[pattern_type]) > 100:
            self._pattern_memory[pattern_type] = self._pattern_memory[pattern_type][-100:]
    
    async def get_similar_patterns(self, pattern_type: str, limit: int = 5) -> list:
        """Get stored patterns of a specific type."""
        patterns = self._pattern_memory.get(pattern_type, [])
        return patterns[-limit:]
    
    # ── Context Retrieval ───────────────────────────────────────────────────
    
    async def get_relevant_context(
        self,
        query: str,
        session_id: str = None,
        caller_id: str = None,
        include_patterns: bool = True
    ) -> str:
        """
        Get relevant context for an assistant query.
        
        Combines:
        - Session context (if session_id provided)
        - Caller history (if caller_id provided)  
        - Relevant patterns (if include_patterns)
        """
        context_parts = []
        
        # Session context
        if session_id:
            session_ctx = await self.get_session_context(session_id)
            if session_ctx:
                context_parts.append(f"Session Data: {json.dumps(session_ctx, default=str)}")
        
        # Caller history
        if caller_id:
            caller_ctx = await self.get_caller_context(caller_id)
            if caller_ctx:
                context_parts.append(f"Caller History:\n{caller_ctx}")
        
        # Pattern context
        if include_patterns:
            # Get relevant violation and obligation patterns
            violations = await self.get_similar_patterns("violations", limit=3)
            if violations:
                context_parts.append(f"Recent violation patterns: {len(violations)} similar cases detected")
        
        if not context_parts:
            return ""
        
        return "\n\n".join(context_parts)
    
    # ── Memory with Backboard ───────────────────────────────────────────────
    
    async def send_with_memory(
        self,
        assistant_id: str,
        content: str,
        session_id: str = None,
        caller_id: str = None,
        memory_mode: str = "Auto"
    ) -> str:
        """
        Send a message to an assistant with memory context.
        
        Args:
            assistant_id: Backboard assistant ID
            content: Message content
            session_id: Optional session for thread persistence
            caller_id: Optional caller for context enrichment
            memory_mode: "Auto", "Full", "Selective", or "None"
        """
        client = await self._get_client()
        
        # Get or create thread
        thread_id = await self.get_or_create_thread(
            assistant_id=assistant_id,
            session_id=session_id,
            caller_id=caller_id,
            use_persistent=(session_id is not None)
        )
        
        # Enrich with context if caller_id provided
        enriched_content = content
        if caller_id:
            caller_context = await self.get_caller_context(caller_id)
            if caller_context:
                enriched_content = f"""CALLER CONTEXT:
{caller_context}

---

{content}"""
        
        # Send to Backboard with memory
        response = await client.add_message(
            thread_id=thread_id,
            content=enriched_content,
            memory=memory_mode,
            stream=False,
        )
        
        return response.content


# ── Module-level instance ───────────────────────────────────────────────────

_memory_manager: Optional[MemoryManager] = None


async def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
        await _memory_manager.initialize()
    return _memory_manager


def generate_session_id() -> str:
    """Generate a unique session ID."""
    import uuid
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
