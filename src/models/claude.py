from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional, Union, Literal

class ClaudeContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ClaudeContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ClaudeContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ClaudeContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]]

class ClaudeContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    text: str = ""
    thinking: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def validate_thinking_field(cls, v):
        """Handle both 'text' and 'thinking' field names for compatibility."""
        if isinstance(v, dict):
            thinking_value = v.get("thinking")
            # If 'thinking' field is present, use it for 'text' field
            if thinking_value is not None:
                v["text"] = thinking_value
            # Ensure at least text field exists
            if "text" not in v and "thinking" not in v:
                v["text"] = ""
        return v

class ClaudeSystemContent(BaseModel):
    type: Literal["text"]
    text: str

class ClaudeMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ClaudeContentBlockText, ClaudeContentBlockImage, ClaudeContentBlockToolUse, ClaudeContentBlockToolResult, ClaudeContentBlockThinking]]]

class ClaudeTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ClaudeThinkingConfig(BaseModel):
    enabled: bool = True
    targeting_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None

class ClaudeMessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[ClaudeTool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ClaudeThinkingConfig] = None

class ClaudeTokenCountRequest(BaseModel):
    model: str
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    tools: Optional[List[ClaudeTool]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
