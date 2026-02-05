import json
import uuid
import logging
from fastapi import HTTPException, Request
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest
from src.core.config import config

logger = logging.getLogger(__name__)


def _accumulate_tool_argument(args_buffer: str, new_args: str) -> str:
    """
    累积工具调用参数，智能处理完整 JSON 与片段。

    Args:
        args_buffer: 当前累积的参数缓冲区
        new_args: 新传入的参数片段或完整 JSON

    Returns:
        更新后的参数缓冲区

    注意事项:
        - 如果 new_args 是完整的有效 JSON，直接替换 args_buffer
        - 如果 new_args 是 JSON 片段，累积到 args_buffer
        - 空片段或 None 会被忽略

    这是处理 OpenAI 流式响应中的特殊情况：
        某些模型可能在最后一块中同时包含 finish_reason="tool_calls" 和工具调用的完整参数，
        此时需要用完整参数替换之前的碎片累积，而不是继续追加。
    """
    # 空值保护
    if not new_args or new_args is None:
        return args_buffer

    # 尝试解析为完整 JSON
    try:
        json.loads(new_args)
        # 如果成功解析，说明是完整 JSON，直接替换
        return new_args
    except json.JSONDecodeError:
        # 解析失败，说明是片段，累积到缓冲区
        return args_buffer + new_args



def convert_openai_to_claude_response(
    openai_response: dict, original_request: ClaudeMessagesRequest
) -> dict:
    """Convert OpenAI response to Claude format."""

    # Extract response data
    choices = openai_response.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="No choices in OpenAI response")

    choice = choices[0]
    message = choice.get("message", {})

    # Build Claude content blocks
    content_blocks = []

    # Add text content first (if present)
    reasoning_content = message.get("reasoning_content")
    if reasoning_content:
        content_blocks.append({"type": Constants.CONTENT_THINKING, "text": reasoning_content})

        # Debug log
        if config.thinking_debug:
            debug_preview = reasoning_content[:100] + "..." if len(reasoning_content) > 100 else reasoning_content
            logger.info(f"[Thinking Response] Preview: {debug_preview}")
            logger.info(f"[Thinking Response] Length: {len(reasoning_content)} chars")

    # Add text content
    text_content = message.get("content")
    if text_content is not None:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": text_content})

    # Add tool calls
    tool_calls = message.get("tool_calls", []) or []
    for tool_call in tool_calls:
        if tool_call.get("type") == Constants.TOOL_FUNCTION:
            function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
            try:
                arguments = json.loads(function_data.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {"raw_arguments": function_data.get("arguments", "")}

            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": tool_call.get("id", f"tool_{uuid.uuid4()}"),
                    "name": function_data.get("name", ""),
                    "input": arguments,
                }
            )

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    # Map finish reason
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = {
        "stop": Constants.STOP_END_TURN,
        "length": Constants.STOP_MAX_TOKENS,
        "tool_calls": Constants.STOP_TOOL_USE,
        "function_call": Constants.STOP_TOOL_USE,
    }.get(finish_reason, Constants.STOP_END_TURN)

    # Build Claude response
    claude_response = {
        "id": openai_response.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get(
                "completion_tokens", 0
            ),
        },
    }

    return claude_response


async def convert_openai_streaming_to_claude(
    openai_stream, original_request: ClaudeMessagesRequest, logger, request_id: str = None
):
    """Convert OpenAI streaming response to Claude streaming format."""

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send initial SSE events
    yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_START, 'message': {'id': message_id, 'type': 'message', 'role': Constants.ROLE_ASSISTANT, 'model': original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_PING}\ndata: {json.dumps({'type': Constants.EVENT_PING}, ensure_ascii=False)}\n\n"

    # Process streaming chunks
    text_block_index = 0
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN

    # Track content block states
    thinking_started = False
    thinking_finished = False
    text_started = False

    try:
        async for line in openai_stream:
            if line.strip():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(chunk_data)
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse chunk: {chunk_data}, error: {e}"
                        )
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Handle reasoning_content (thinking) delta
                    if delta and "reasoning_content" in delta and delta["reasoning_content"] is not None and delta["reasoning_content"] != "":
                        # 首次遇到非空 reasoning_content，发送 start 事件
                        if not thinking_started:
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': 0, 'content_block': {'type': Constants.CONTENT_THINKING}}, ensure_ascii=False)}\n\n"
                            thinking_started = True

                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': 0, 'delta': {'type': Constants.DELTA_THINKING, 'thinking': delta['reasoning_content']}}, ensure_ascii=False)}\n\n"

                        # Debug log
                        if config.thinking_debug:
                            logger.debug(f"[Thinking Stream Delta] Adding: {delta['reasoning_content'][:50]}...")

                    # Handle text delta
                    if delta and "content" in delta and delta["content"] is not None and delta["content"] != "":
                        # 首次遇到非空文本内容，根据是否有 thinking 确定索引
                        if not text_started:
                            # 如果 thinking 已启动且未结束，先发送 thinking stop
                            if thinking_started and not thinking_finished:
                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0}, ensure_ascii=False)}\n\n"
                                thinking_finished = True

                            text_block_index = 1 if thinking_started else 0
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': text_block_index, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}}, ensure_ascii=False)}\n\n"
                            text_started = True

                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': delta['content']}}, ensure_ascii=False)}\n\n"

                    # Handle tool call deltas with improved incremental processing
                    if "tool_calls" in delta:
                        # 如果 thinking 已启动且未结束，先发送 thinking stop（第一个 tool call）
                        if thinking_started and not thinking_finished:
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0}, ensure_ascii=False)}\n\n"
                            text_finished = True

                        for tc_delta in delta["tool_calls"]:
                            tc_index = tc_delta.get("index", 0)

                            # Initialize tool call tracking by index if not exists
                            if tc_index not in current_tool_calls:
                                current_tool_calls[tc_index] = {
                                    "id": None,
                                    "name": None,
                                    "args_buffer": "",
                                    "sent": False,
                                    "claude_index": None,
                                    "started": False
                                }

                            tool_call = current_tool_calls[tc_index]

                            # Update tool call ID if provided
                            if tc_delta.get("id"):
                                tool_call["id"] = tc_delta["id"]

                            # Update function name and start content block if we have both id and name
                            function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
                            if function_data.get("name"):
                                tool_call["name"] = function_data["name"]

                            # Start content block when we have complete initial data
                            if (tool_call["id"] and tool_call["name"] and not tool_call["started"]):
                                tool_block_counter += 1
                                claude_index = text_block_index + tool_block_counter
                                tool_call["claude_index"] = claude_index
                                tool_call["started"] = True

                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': claude_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_call['id'], 'name': tool_call['name'], 'input': {}}}, ensure_ascii=False)}\n\n"

                            # Handle function arguments - accumulate without sending intermediate events
                            if "arguments" in function_data and tool_call["started"] and function_data["arguments"] is not None:
                                tool_call["args_buffer"] = _accumulate_tool_argument(
                                    tool_call["args_buffer"], function_data["arguments"]
                                )
                                # Note: Complete JSON will be sent when finish_reason is tool_calls

                    # Handle finish reason
                    if finish_reason:
                        # 如果 thinking 已启动且未结束，先发送 thinking stop
                        if thinking_started and not thinking_finished:
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0}, ensure_ascii=False)}\n\n"
                            text_finished = True

                        # 发送完整的工具调用（对于流式或旧版不流式情况）
                        if finish_reason in ["tool_calls", "function_call"]:
                            for tool_data in current_tool_calls.values():
                                if tool_data.get("started") and tool_data.get("args_buffer") and not tool_data.get("sent"):
                                    try:
                                        # Validate JSON before sending
                                        json.loads(tool_data["args_buffer"])
                                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': tool_data['claude_index'], 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': tool_data['args_buffer']}}, ensure_ascii=False)}\n\n"
                                        tool_data["sent"] = True
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse tool arguments: {tool_data['args_buffer'][:100]}..., error: {e}")

                        if finish_reason == "length":
                            final_stop_reason = Constants.STOP_MAX_TOKENS
                        elif finish_reason in ["tool_calls", "function_call"]:
                            final_stop_reason = Constants.STOP_TOOL_USE
                        elif finish_reason == "stop":
                            final_stop_reason = Constants.STOP_END_TURN
                        else:
                            final_stop_reason = Constants.STOP_END_TURN
                        break

    except Exception as e:
        # Handle any streaming errors gracefully
        logger.error(f"Streaming error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
        }
        yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        return

    # Send final SSE events
    # 如果有 thinking content 且未结束，先发送其 stop 事件
    if thinking_started and not thinking_finished:
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0}, ensure_ascii=False)}\n\n"
        text_finished = True

    # 如果有 text content，发送其 stop 事件
    if text_started:
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': text_block_index}, ensure_ascii=False)}\n\n"

    # 发送未发送的工具调用（用于新版不流式情况）
    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("args_buffer") and not tool_data.get("sent"):
            try:
                json.loads(tool_data["args_buffer"])  # Validate
                yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': tool_data['claude_index'], 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': tool_data['args_buffer']}}, ensure_ascii=False)}\n\n"
                tool_data["sent"] = True
            except json.JSONDecodeError:
                pass  # JSON invalid, skip

    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': tool_data['claude_index']}, ensure_ascii=False)}\n\n"

    usage_data = {"input_tokens": 0, "output_tokens": 0}
    yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': usage_data}, ensure_ascii=False)}\n\n"
    yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP}, ensure_ascii=False)}\n\n"
    logger.info(f"Claude流式事件转换完成 (request_id={request_id})")


async def convert_openai_streaming_to_claude_with_cancellation(
    openai_stream,
    original_request: ClaudeMessagesRequest,
    logger,
    http_request: Request,
    openai_client,
    request_id: str,
):
    """Convert OpenAI streaming response to Claude streaming format with cancellation support."""

    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    event_count = 0

    # Send initial SSE events
    yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_START, 'message': {'id': message_id, 'type': 'message', 'role': Constants.ROLE_ASSISTANT, 'model': original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}}, ensure_ascii=False)}\n\n"

    # Don't send initial content_block_start yet - we need to determine if there will be text
    yield f"event: {Constants.EVENT_PING}\ndata: {json.dumps({'type': Constants.EVENT_PING}, ensure_ascii=False)}\n\n"

    # Process streaming chunks
    text_block_index = 0
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN
    usage_data = {"input_tokens": 0, "output_tokens": 0}

    # Track content block states
    text_started = False
    thinking_started = False
    thinking_finished = False

    try:
        async for line in openai_stream:
            # Check if client disconnected
            if await http_request.is_disconnected():
                logger.info(f"Client disconnected, cancelling request {request_id}")
                openai_client.cancel_request(request_id)
                break

            if line.strip():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(chunk_data)
                        # logger.info(f"OpenAI chunk: {chunk}")
                        usage = chunk.get("usage", None)
                        if usage:
                            cache_read_input_tokens = 0
                            prompt_tokens_details = usage.get('prompt_tokens_details', {})
                            if prompt_tokens_details:
                                cache_read_input_tokens = prompt_tokens_details.get('cached_tokens', 0)
                            usage_data = {
                                'input_tokens': usage.get('prompt_tokens', 0),
                                'output_tokens': usage.get('completion_tokens', 0),
                                'cache_read_input_tokens': cache_read_input_tokens
                            }
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse chunk: {chunk_data}, error: {e}"
                        )
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Handle reasoning_content (thinking) delta
                    if delta and "reasoning_content" in delta and delta["reasoning_content"] is not None and delta["reasoning_content"] != "":
                        # 首次遇到非空 reasoning_content，发送 start 事件
                        if not thinking_started:
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': 0, 'content_block': {'type': Constants.CONTENT_THINKING}}, ensure_ascii=False)}\n\n"
                            thinking_started = True

                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': 0, 'delta': {'type': Constants.DELTA_THINKING, 'thinking': delta['reasoning_content']}}, ensure_ascii=False)}\n\n"

                        # Debug log
                        if config.thinking_debug:
                            logger.debug(f"[Thinking Stream Delta] Adding: {delta['reasoning_content'][:50]}...")

                    # Handle text delta
                    if delta and "content" in delta and delta["content"] is not None and delta["content"] != "":
                        # 首次遇到非空文本内容，根据是否有 text 确定索引
                        if not text_started:
                            # 如果 thinking 已启动且未结束，先发送 text stop
                            if thinking_started and not thinking_finished:
                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0}, ensure_ascii=False)}\n\n"
                                thinking_finished = True

                            text_block_index = 1 if thinking_started else 0
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': text_block_index, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}}, ensure_ascii=False)}\n\n"
                            text_started = True

                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': delta['content']}}, ensure_ascii=False)}\n\n"

                    # Handle tool call deltas with improved incremental processing
                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc_delta in delta["tool_calls"]:
                            tc_index = tc_delta.get("index", 0)

                            # Initialize tool call tracking by index if not exists
                            if tc_index not in current_tool_calls:
                                current_tool_calls[tc_index] = {
                                    "id": None,
                                    "name": None,
                                    "args_buffer": "",
                                    "sent": False,
                                    "claude_index": None,
                                    "started": False
                                }

                            tool_call = current_tool_calls[tc_index]

                            # Update tool call ID if provided
                            if tc_delta.get("id"):
                                tool_call["id"] = tc_delta["id"]

                            # Update function name and start content block if we have both id and name
                            function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
                            if function_data.get("name"):
                                tool_call["name"] = function_data["name"]
                            
                            # Start content block when we have complete initial data
                            if (tool_call["id"] and tool_call["name"] and not tool_call["started"]):
                                tool_block_counter += 1
                                claude_index = text_block_index + tool_block_counter
                                tool_call["claude_index"] = claude_index
                                tool_call["started"] = True
                                
                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': claude_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_call['id'], 'name': tool_call['name'], 'input': {}}}, ensure_ascii=False)}\n\n"
                            
                            # Handle function arguments - accumulate without sending intermediate events
                            if "arguments" in function_data and tool_call["started"] and function_data["arguments"] is not None:
                                tool_call["args_buffer"] = _accumulate_tool_argument(
                                    tool_call["args_buffer"], function_data["arguments"]
                                )
                                # Note: Complete JSON will be sent when finish_reason is tool_calls

                    # Handle finish reason
                    if finish_reason:
                        # 发送完整的工具调用（对于流式或旧版不流式情况）
                        if finish_reason in ["tool_calls", "function_call"]:
                            for tool_data in current_tool_calls.values():
                                if tool_data.get("started") and tool_data.get("args_buffer") and not tool_data.get("sent"):
                                    try:
                                        # Validate JSON before sending
                                        json.loads(tool_data["args_buffer"])
                                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': tool_data['claude_index'], 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': tool_data['args_buffer']}}, ensure_ascii=False)}\n\n"
                                        tool_data["sent"] = True
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse tool arguments: {tool_data['args_buffer'][:100]}..., error: {e}")

                        if finish_reason == "length":
                            final_stop_reason = Constants.STOP_MAX_TOKENS
                        elif finish_reason in ["tool_calls", "function_call"]:
                            final_stop_reason = Constants.STOP_TOOL_USE
                        elif finish_reason == "stop":
                            final_stop_reason = Constants.STOP_END_TURN
                        else:
                            final_stop_reason = Constants.STOP_END_TURN

    except HTTPException as e:
        # Handle cancellation
        if e.status_code == 499:
            logger.info(f"Request {request_id} was cancelled")
            error_event = {
                "type": "error",
                "error": {
                    "type": "cancelled",
                    "message": "Request was cancelled by client",
                },
            }
            yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            return
        else:
            raise
    except Exception as e:
        # Handle any streaming errors gracefully
        logger.error(f"Streaming error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
        }
        yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        return

    # Send final SSE events
    # 如果有 thinking content 且未结束，先发送其 stop 事件
    if thinking_started and not thinking_finished:
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0}, ensure_ascii=False)}\n\n"
        thinking_finished = True

    # 如果有 text content，发送其 stop 事件
    if text_started:
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': text_block_index}, ensure_ascii=False)}\n\n"

    # 发送未发送的工具调用（用于新版不流式情况）
    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("args_buffer") and not tool_data.get("sent"):
            try:
                json.loads(tool_data["args_buffer"])  # Validate
                yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': tool_data['claude_index'], 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': tool_data['args_buffer']}}, ensure_ascii=False)}\n\n"
                tool_data["sent"] = True
            except json.JSONDecodeError:
                pass  # JSON invalid, skip

    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {json.dumps({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': tool_data['claude_index']}, ensure_ascii=False)}\n\n"

    yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': usage_data}, ensure_ascii=False)}\n\n"
    yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {json.dumps({'type': Constants.EVENT_MESSAGE_STOP}, ensure_ascii=False)}\n\n"
    logger.info(f"Claude流式事件转换完成 (request_id={request_id})")
