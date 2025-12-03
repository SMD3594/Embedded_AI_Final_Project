# kv_cache_manager.py
# Phase 2.3: Custom Paged and Rotating KV Cache data structures

class PagedKVCache:
    def __init__(self, page_size, max_pages, key_dim=None, value_dim=None):
        """
        A paged KV cache that keeps tokens in fixed-size pages.
        When max_pages is reached, the oldest page is dropped.
        """
        self.page_size = page_size
        self.max_pages = max_pages
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Each page is a list of (K, V) pairs
        self.cached_K_pages = []
        self.cached_V_pages = []

    def append(self, K_new, V_new):
        """
        Append a new (K, V) pair. If the current page is full,
        start a new page. If we've hit max_pages, drop the oldest page.
        """
        # Start a new page if none exist or the last page is full
        if not self.cached_K_pages or len(self.cached_K_pages[-1]) >= self.page_size:
            # If at max pages, remove the oldest one
            if len(self.cached_K_pages) >= self.max_pages:
                self.cached_K_pages.pop(0)
                self.cached_V_pages.pop(0)

            # Add a new empty page
            self.cached_K_pages.append([])
            self.cached_V_pages.append([])

        # Append K_new and V_new to the current (last) page
        self.cached_K_pages[-1].append(K_new)
        self.cached_V_pages[-1].append(V_new)

    def get_recent(self, n):
        """
        Return the most recent n (K, V) entries across pages,
        ordered from oldest to newest.
        """
        recent_K = []
        recent_V = []
        remaining = n

        for page_idx in range(len(self.cached_K_pages) - 1, -1, -1):
            if remaining <= 0:
                break

            K_page = self.cached_K_pages[page_idx]
            V_page = self.cached_V_pages[page_idx]

            take = min(remaining, len(K_page))

            recent_K.extend(K_page[-take:])
            recent_V.extend(V_page[-take:])

            remaining -= take

        # reverse from newest to oldest; oldest to newest
        recent_K.reverse()
        recent_V.reverse()

        return recent_K, recent_V

    def reset(self):
        """Clear all pages."""
        self.cached_K_pages = []
        self.cached_V_pages = []


class RotatingKVCache:
    def __init__(self, max_tokens, key_dim=None, value_dim=None):
        """
        A fixed-size circular buffer for KV pairs.
        Once full, it overwrites the oldest entries.
        """
        self.max_tokens = max_tokens
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.cached_K = [None] * max_tokens
        self.cached_V = [None] * max_tokens

        self.current_index = 0  # next write position
        self.size = 0           # number of valid entries
    
    def append(self, K_new, V_new):
        """
        Store (K_new, V_new) at the current index, then advance.
        """
        self.cached_K[self.current_index] = K_new
        self.cached_V[self.current_index] = V_new

        self.current_index = (self.current_index + 1) % self.max_tokens
        
        # Update size; can't exceed max_tokens
        self.size = min(self.size + 1, self.max_tokens)

    def get_recent(self, n):
        """
        Return the most recent n entries (up to self.size), from oldest to newest.
        """
        n = min(n, self.size)

        recent_K = []
        recent_V = []
        
        for i in range(n):
            idx = (self.current_index - 1 - i) % self.max_tokens
            recent_K.append(self.cached_K[idx])
            recent_V.append(self.cached_V[idx])
        
        recent_K.reverse()
        recent_V.reverse()
        
        return recent_K, recent_V

    def reset(self):
        """Clear the rotating buffer."""
        self.cached_K = [None] * self.max_tokens
        self.cached_V = [None] * self.max_tokens
        self.current_index = 0
        self.size = 0


if __name__ == "__main__":
    # --Optional-- for debugging
    print("=== Testing PagedKVCache ===")
    paged_cache = PagedKVCache(page_size=3, max_pages=2)

    for i in range(8):
        paged_cache.append(f"K{i}", f"V{i}")

    print("K pages:", paged_cache.cached_K_pages)
    print("V pages:", paged_cache.cached_V_pages)
    print("Paged recent 3:", paged_cache.get_recent(3))
    print("Paged recent 5:", paged_cache.get_recent(5))

    print("\n=== Testing RotatingKVCache ===")
    rotating_cache = RotatingKVCache(max_tokens=5)

    for i in range(8):   # more than max_tokens → will wrap around
        rotating_cache.append(f"K{i}", f"V{i}")

    print("Full rotating K buffer:", rotating_cache.cached_K)
    print("current_index:", rotating_cache.current_index)
    print("size:", rotating_cache.size)
    print("Rotating recent 3:", rotating_cache.get_recent(3))
    print("Rotating recent 5:", rotating_cache.get_recent(5))
