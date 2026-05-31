import httpx


class HermesClient:
    """HTTP client for Hermes API Server.

    Maintains a multi-turn conversation context in ``messages``.
    """

    def __init__(self, base_url, api_key, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.messages = []
        self._client = httpx.Client(timeout=httpx.Timeout(timeout))

    def send(self, text):
        """Send user *text*, get assistant reply.

        Appends the user message to the conversation context before
        sending, then appends the assistant reply on success.
        """
        self.messages.append({"role": "user", "content": text})

        payload = {
            "model": "hermes-agent",
            "messages": list(self.messages),
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            self.messages.pop()
            raise ConnectionError(
                f"Hermes API request failed: {exc}"
            ) from exc

        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def clear_context(self):
        self.messages.clear()

    def close(self):
        self._client.close()
