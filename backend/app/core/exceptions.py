"""Typed errors for every layer.

The rule: a failure anywhere below the API layer raises an ``OmnivoresError``
subclass carrying a stable ``code`` and a ``user_message``. The WebSocket and
REST layers translate those into structured error events instead of dropping
the connection or leaking tracebacks. A dead vLLM server must never look like a
hung UI.
"""

from __future__ import annotations

from typing import Any


class OmnivoresError(Exception):
    """Base class for all application errors."""

    code = "internal_error"
    http_status = 500
    #: Safe to show to an end user verbatim.
    user_message = "Something went wrong."
    #: True when retrying the same request could plausibly succeed.
    retryable = False

    def __init__(
        self,
        message: str | None = None,
        *,
        user_message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or self.user_message)
        self.message = message or self.user_message
        if user_message is not None:
            self.user_message = user_message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.user_message,
            "detail": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


# --- Configuration ---------------------------------------------------------


class ConfigurationError(OmnivoresError):
    code = "configuration_error"
    user_message = "The server is misconfigured."


# --- Inference -------------------------------------------------------------


class InferenceError(OmnivoresError):
    code = "inference_error"
    http_status = 502
    user_message = "The language model could not be reached."


class InferenceUnavailable(InferenceError):
    code = "inference_unavailable"
    http_status = 503
    user_message = "AI inference is currently unavailable."
    retryable = True


class ModelNotLoaded(InferenceError):
    code = "model_not_loaded"
    http_status = 503
    user_message = "The model is still loading. Try again in a moment."
    retryable = True


class InferenceTimeout(InferenceError):
    code = "inference_timeout"
    http_status = 504
    user_message = "The model took too long to respond."
    retryable = True


class ContextLengthExceeded(InferenceError):
    code = "context_length_exceeded"
    http_status = 413
    user_message = "This conversation is too long for the model's context window."


# --- Tools -----------------------------------------------------------------


class ToolError(OmnivoresError):
    code = "tool_error"
    user_message = "A tool failed to run."


class UnknownTool(ToolError):
    code = "unknown_tool"
    user_message = "The agent requested a tool that does not exist."


class InvalidToolArguments(ToolError):
    code = "invalid_tool_arguments"
    http_status = 400
    user_message = "The agent called a tool with invalid arguments."


class ToolExecutionError(ToolError):
    code = "tool_execution_error"
    user_message = "A tool failed while running."


class ToolTimeout(ToolError):
    code = "tool_timeout"
    http_status = 504
    user_message = "A tool took too long and was cancelled."
    retryable = True


class ToolPermissionDenied(ToolError):
    code = "tool_permission_denied"
    http_status = 403
    user_message = "This agent is not allowed to use that tool."


# --- Sandbox ---------------------------------------------------------------


class SandboxError(ToolError):
    code = "sandbox_error"
    user_message = "Code execution failed."


class SandboxDisabled(SandboxError):
    code = "sandbox_disabled"
    user_message = (
        "Code execution is disabled on this server. Set PYTHON_SANDBOX_MODE=docker "
        "to enable it."
    )


class SandboxUnavailable(SandboxError):
    code = "sandbox_unavailable"
    http_status = 503
    user_message = "The code execution sandbox is not available."
    retryable = True


class SandboxTimeout(SandboxError):
    code = "sandbox_timeout"
    user_message = "Code execution exceeded the time limit."


# --- Documents / RAG -------------------------------------------------------


class DocumentError(OmnivoresError):
    code = "document_error"
    http_status = 400
    user_message = "The document could not be processed."


class UnsupportedFileType(DocumentError):
    code = "unsupported_file_type"
    user_message = "That file type is not supported."


class FileTooLarge(DocumentError):
    code = "file_too_large"
    http_status = 413
    user_message = "That file is too large."


class DocumentNotFound(DocumentError):
    code = "document_not_found"
    http_status = 404
    user_message = "Document not found."


class DocumentParseError(DocumentError):
    code = "document_parse_error"
    http_status = 422
    user_message = "The document could not be read."


class EmbeddingError(OmnivoresError):
    code = "embedding_error"
    user_message = "Failed to generate embeddings."


class VectorStoreError(OmnivoresError):
    code = "vector_store_error"
    user_message = "The document index is unavailable."


class RetrievalError(OmnivoresError):
    code = "retrieval_error"
    user_message = "Document search failed."


# --- Agents / channels -----------------------------------------------------


class AgentError(OmnivoresError):
    code = "agent_error"
    user_message = "The agent failed."


class UnknownAgent(AgentError):
    code = "unknown_agent"
    http_status = 404
    user_message = "No such agent."


class AgentTimeout(AgentError):
    code = "agent_timeout"
    http_status = 504
    user_message = "The agent took too long and was stopped."
    retryable = True


class ChannelNotFound(OmnivoresError):
    code = "channel_not_found"
    http_status = 404
    user_message = "Channel not found."


# --- Transport / security --------------------------------------------------


class AuthenticationError(OmnivoresError):
    code = "authentication_error"
    http_status = 401
    user_message = "Authentication required."


class RateLimitExceeded(OmnivoresError):
    code = "rate_limit_exceeded"
    http_status = 429
    user_message = "You are sending messages too quickly."
    retryable = True


class ValidationError(OmnivoresError):
    code = "validation_error"
    http_status = 400
    user_message = "The request was invalid."
