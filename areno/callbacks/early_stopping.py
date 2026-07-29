from typing import Dict, Any

class EarlyStopping:
    def __init__(self, monitor="eval_loss", patience=3, mode="min", min_delta=0.0, verbose=True):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        if mode == "min":
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta

    def __call__(self, metrics: Dict[str, Any]) -> bool:
        if self.monitor not in metrics:
            return False
        current_score = metrics[self.monitor]
        if self.best_score is None:
            self.best_score = current_score
            return False
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] {self.monitor} improved to {current_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] {self.monitor} did not improve. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False

    def state_dict(self) -> Dict[str, Any]:
        return {"counter": self.counter, "best_score": self.best_score, "early_stop": self.early_stop}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
        self.early_stop = state_dict["early_stop"]
