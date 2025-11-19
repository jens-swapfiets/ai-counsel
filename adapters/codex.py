"""Codex CLI adapter."""
import asyncio
import logging
from typing import Optional

from adapters.base import BaseCLIAdapter

logger = logging.getLogger(__name__)


class CodexAdapter(BaseCLIAdapter):
    """Adapter for codex CLI tool."""

    # Model-specific reasoning effort mapping
    # Maps model IDs to their supported reasoning effort levels
    REASONING_EFFORT_MAP = {
        "gpt-5.1-codex-max": "xhigh",   # Extended reasoning with xhigh support
        "gpt-5.1-codex": "high",        # Standard model, max is 'high'
        "gpt-5.1-codex-mini": "medium", # Mini variant, supports up to 'high'
        "gpt-5.1": "medium",            # General model, supports up to 'high'
        "gpt-5-codex": "high",          # Legacy model
        "gpt-5": "high",                # Legacy model
    }

    def __init__(
        self, command: str = "codex", args: list[str] | None = None, timeout: int = 60
    ):
        """
        Initialize Codex adapter.

        Args:
            command: Command to execute (default: "codex")
            args: List of argument templates (from config.yaml)
            timeout: Timeout in seconds (default: 60)

        Note:
            The codex CLI uses `codex exec "prompt"` syntax.
            Model is configured via ~/.codex/config.toml, not passed as CLI arg.
        """
        if args is None:
            raise ValueError("args must be provided from config.yaml")
        super().__init__(command=command, args=args, timeout=timeout)

    async def invoke(
        self,
        prompt: str,
        model: str,
        context: Optional[str] = None,
        is_deliberation: bool = True,
        working_directory: Optional[str] = None,
    ) -> str:
        """
        Invoke the Codex CLI with model-aware reasoning effort adjustment.

        Automatically selects the appropriate reasoning effort level based on
        the model being used, preventing API errors when using models that
        don't support certain reasoning effort levels.

        Args:
            prompt: The prompt to send to the model
            model: Model identifier (e.g., "gpt-5.1-codex", "gpt-5.1-codex-max")
            context: Optional additional context
            is_deliberation: Whether this is part of a deliberation
            working_directory: Optional working directory for subprocess execution

        Returns:
            Parsed response from the model

        Raises:
            TimeoutError: If execution exceeds timeout
            RuntimeError: If CLI process fails
        """
        # Determine appropriate reasoning effort for this model
        reasoning_effort = self.REASONING_EFFORT_MAP.get(model, "high")
        logger.info(
            f"Codex adapter: using reasoning effort '{reasoning_effort}' for model '{model}'"
        )

        # Build full prompt
        full_prompt = prompt
        if context:
            full_prompt = f"{context}\n\n{prompt}"

        # Validate prompt length if adapter supports it
        if hasattr(self, "validate_prompt_length"):
            if not self.validate_prompt_length(full_prompt):
                raise ValueError(
                    f"Prompt too long ({len(full_prompt)} chars). "
                    f"Maximum allowed: {getattr(self, 'MAX_PROMPT_CHARS', 'unknown')} chars. "
                    "This prevents API rejection errors."
                )

        # Adjust args based on context (for auto-detecting deliberation mode)
        args = self._adjust_args_for_context(is_deliberation)

        # Determine working directory for subprocess
        import os

        cwd = working_directory if working_directory else os.getcwd()

        # Format arguments with {model}, {prompt}, and {working_directory} placeholders
        formatted_args = [
            arg.format(model=model, prompt=full_prompt, working_directory=cwd)
            for arg in args
        ]

        # Inject reasoning effort via config override if not already present
        # The codex CLI uses `-c` flag for config overrides with dotted path notation
        config_override = f"model_reasoning_effort={reasoning_effort}"
        if "-c" not in formatted_args and config_override not in formatted_args:
            # Insert after "exec" command but before other flags
            if "exec" in formatted_args:
                exec_index = formatted_args.index("exec")
                formatted_args.insert(exec_index + 1, "-c")
                formatted_args.insert(exec_index + 2, config_override)
            else:
                # Fallback: add at the beginning
                formatted_args.insert(0, "-c")
                formatted_args.insert(1, config_override)

        # Log the command being executed
        logger.info(
            f"Executing Codex CLI adapter: command={self.command}, "
            f"model={model}, reasoning_effort={reasoning_effort}, cwd={cwd}, "
            f"prompt_length={len(full_prompt)} chars"
        )
        logger.debug(f"Full command: {self.command} {' '.join(formatted_args[:5])}... (args truncated)")

        # Execute with retry logic for transient errors
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                process = await asyncio.create_subprocess_exec(
                    self.command,
                    *formatted_args,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.timeout
                )

                if process.returncode != 0:
                    error_msg = stderr.decode("utf-8", errors="replace")

                    # Check if this is a transient error
                    is_transient = self._is_transient_error(error_msg)

                    if is_transient and attempt < self.max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(
                            f"Transient error detected (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg[:100]}. "
                            f"Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                        last_error = error_msg
                        continue

                    logger.error(
                        f"Codex CLI process failed: command={self.command}, "
                        f"model={model}, reasoning_effort={reasoning_effort}, "
                        f"returncode={process.returncode}, "
                        f"error={error_msg[:200]}"
                    )
                    raise RuntimeError(f"CLI process failed: {error_msg}")

                raw_output = stdout.decode("utf-8", errors="replace")
                if attempt > 0:
                    logger.info(
                        f"Codex CLI adapter succeeded on retry attempt {attempt + 1}: "
                        f"command={self.command}, model={model}"
                    )
                logger.info(
                    f"Codex CLI adapter completed successfully: command={self.command}, "
                    f"model={model}, reasoning_effort={reasoning_effort}, "
                    f"output_length={len(raw_output)} chars"
                )
                logger.debug(f"Raw output preview: {raw_output[:500]}...")
                return self.parse_output(raw_output)

            except asyncio.TimeoutError:
                logger.error(
                    f"Codex CLI invocation timed out: command={self.command}, "
                    f"model={model}, reasoning_effort={reasoning_effort}, timeout={self.timeout}s"
                )
                raise TimeoutError(f"CLI invocation timed out after {self.timeout}s")

        # All retries exhausted
        raise RuntimeError(f"CLI failed after {self.max_retries + 1} attempts. Last error: {last_error}")

    def parse_output(self, raw_output: str) -> str:
        """
        Parse codex output.

        Codex outputs clean responses without header/footer text,
        so we simply strip whitespace.

        Args:
            raw_output: Raw stdout from codex

        Returns:
            Parsed model response
        """
        return raw_output.strip()
