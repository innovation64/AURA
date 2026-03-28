"""Three interaction paradigms for agent-environment communication.

- Reactive:      Agent queries environment only when it needs information.
- Proactive:     Environment pushes relevant context to agent before it asks.
- Collaborative: Proactive + online feedback loop that adapts push quality.
"""

from .reactive import ReactiveParadigm
from .proactive import ProactiveParadigm
from .collaborative import CollaborativeParadigm

__all__ = ["ReactiveParadigm", "ProactiveParadigm", "CollaborativeParadigm"]
