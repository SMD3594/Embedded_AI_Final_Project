class slidingWindowCache:

  def __init__(self, window_size: int): #Establishes the caches max size
      """Initializes the cache's max size."""
      if not isinstance(window_size, int):
          raise TypeError("Window size must be an integer")

      self.window_size = window_size
      self._window = []

  def add_token(self, token: any): #Adds a new token at the end of the window.
      """Adds a new token to the end of the window."""
      self._window.append(token)

      if len(self._window) > self.window_size:
          self._window.pop(0)

  def get_tokens(self) -> list: #Returns all tokens in the sliding window.
      """Returns all tokens currently in SW from oldest to newest one."""
      return self._window

  def is_full(self) -> bool: #Checks if the chache is full.
      """Checks if the cache is at its max."""
      return len(self._window) == self.window_size

  def __len__(self): #Returns the current number of tokens in the chahce
      return len(self._window)

  def __repr__(self):
      return f"SlidingWindowCache(size={len(self._window)}/{self.window_size}, tokens={self._window})"
