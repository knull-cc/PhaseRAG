from PhaseRAG.models.phase_memory import RaftPhaseMemory, build_raft_memory
from PhaseRAG.models.phase_retriever import RaftRetriever
from PhaseRAG.models.phase_tokenizer import PhaseTokenizer
from PhaseRAG.models.phaseformer import PhaseFormer

__all__ = [
    "PhaseFormer",
    "PhaseRAGForecaster",
    "PhaseTokenizer",
    "RaftPhaseMemory",
    "RaftRetriever",
    "build_raft_memory",
]


def __getattr__(name: str) -> object:
    if name == "PhaseRAGForecaster":
        from PhaseRAG.models.phase_rag_forecaster import PhaseRAGForecaster

        return PhaseRAGForecaster
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
