from PhaseRAG.models.phase_memory import PhaseMemoryBank, build_phase_memory_bank
from PhaseRAG.models.phase_retriever import PhaseRetriever
from PhaseRAG.models.phase_tokenizer import PhaseTokenizer

__all__ = [
    "PhaseMemoryBank",
    "PhaseRAGForecaster",
    "PhaseRetriever",
    "PhaseTokenizer",
    "build_phase_memory_bank",
]


def __getattr__(name: str) -> object:
    if name == "PhaseRAGForecaster":
        from PhaseRAG.models.phase_rag_forecaster import PhaseRAGForecaster

        return PhaseRAGForecaster
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
