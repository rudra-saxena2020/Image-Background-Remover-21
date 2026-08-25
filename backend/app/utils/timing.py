import time
from dataclasses import dataclass, field


@dataclass
class TimingContext:
    stages: dict[str, float] = field(default_factory=dict)
    _starts: dict[str, float] = field(default_factory=dict)

    def start(self, stage: str) -> None:
        self._starts[stage] = time.perf_counter()

    def end(self, stage: str) -> None:
        if stage in self._starts:
            self.stages[stage] = (time.perf_counter() - self._starts.pop(stage)) * 1000

    def total_ms(self) -> float:
        return sum(self.stages.values())

    def log(self) -> str:
        lines = [f"  {k}: {v:.0f}ms" for k, v in self.stages.items()]
        lines.append(f"  Total: {self.total_ms():.0f}ms")
        return "\n".join(lines)
