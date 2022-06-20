class Preprocessor:
  current_folder: str

  def __init__(self) -> None:
    raise RuntimeError("default Preprocessor not implemented.")

  def reset(self) -> None:
    raise RuntimeError("default Preprocessor not implemented.")

  def preprocess(self, observation) -> None:
    raise RuntimeError("default Preprocessor not implemented.")
