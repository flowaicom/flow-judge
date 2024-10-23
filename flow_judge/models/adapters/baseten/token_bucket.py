import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """Implements a token bucket algorithm for rate limiting.

    This class manages a token bucket with a specified capacity and fill rate,
    allowing for controlled consumption of tokens over time.

    Attributes:
        tokens (float): Current number of tokens in the bucket.
        fill_rate (float): Rate at which tokens are added to the bucket (tokens per second).
        capacity (float): Maximum number of tokens the bucket can hold.
        last_update (float): Timestamp of the last token update.

    Note:
        This implementation is not thread-safe. If used in a multi-threaded environment,
        external synchronization mechanisms should be applied.
    """

    tokens: float
    fill_rate: float
    capacity: float
    last_update: float = field(default_factory=time.time)

    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from the bucket.

        Args:
            tokens (int): Number of tokens to consume. Defaults to 1.

        Returns:
            bool: True if tokens were successfully consumed, False otherwise.

        Note:
            This method updates the token count based on the time elapsed since
            the last update, then attempts to consume the requested number of tokens.
        """
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + self.fill_rate * (now - self.last_update))
        self.last_update = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
