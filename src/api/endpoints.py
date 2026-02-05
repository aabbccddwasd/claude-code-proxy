from fastapi import APIRouter, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
import json
import uuid
from typing import Optional

from src.core.config import config
from src.core.logging import logger
from src.core.client import OpenAIClient
from src.models.claude import ClaudeMessagesRequest, ClaudeTokenCountRequest, EventLoggingBatchResponse
from src.conversion.request_converter import convert_claude_to_openai
from src.conversion.response_converter import (
    convert_openai_to_claude_response,
    convert_openai_streaming_to_claude_with_cancellation,
)
from src.core.model_manager import model_manager

router = APIRouter()

# Get custom headers from config
custom_headers = config.get_custom_headers()

openai_client = OpenAIClient(
    config.openai_api_key,
    config.openai_base_url,
    config.request_timeout,
    api_version=config.azure_api_version,
    custom_headers=custom_headers,
)

async def validate_api_key(x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    """Validate the client's API key from either x-api-key header or Authorization header."""
    client_api_key = None
    
    # Extract API key from headers
    if x_api_key:
        client_api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        client_api_key = authorization.replace("Bearer ", "")
    
    # Skip validation if ANTHROPIC_API_KEY is not set in the environment
    if not config.anthropic_api_key:
        return
        
    # Validate the client API key
    if not client_api_key or not config.validate_client_api_key(client_api_key):
        logger.warning(f"Invalid API key provided by client")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please provide a valid Anthropic API key."
        )

@router.post("/v1/messages")
async def create_message(request: ClaudeMessagesRequest, http_request: Request, _: None = Depends(validate_api_key)):
    try:
        logger.debug(
            f"Processing Claude request: model={request.model}, stream={request.stream}"
        )

        # Generate unique request ID for cancellation tracking
        request_id = str(uuid.uuid4())

        # Log原始Anthropic请求（排除tools和system）
        claude_request_log = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stop_sequences": request.stop_sequences,
            "tools_count": len(request.tools) if request.tools else 0,
            "tool_choice": request.tool_choice,
            "thinking": {
                "enabled": request.thinking.enabled if request.thinking else None,
                "thinking_budget": request.thinking.thinking_budget if request.thinking else None,
            } if request.thinking else None,
            "messages_count": len(request.messages),
            "messages_preview": [
                {"role": msg.role, "content_preview": str(msg.content)[:200] if msg.content else None}
                for msg in request.messages[:3]
            ],
        }
        logger.info(f"[Request {request_id}] Claude原始请求: {json.dumps(claude_request_log, ensure_ascii=False)}")

        # Convert Claude request to OpenAI format (returns tuple)
        openai_request, extra_body = convert_claude_to_openai(request, model_manager)

        # Log转换后的OpenAI请求（过滤掉tools和system）
        # 过滤 messages 中的 system 消息，只保留数量
        filtered_messages = []
        system_count = 0
        for msg in openai_request.get("messages", []):
            if msg.get("role") == "system":
                system_count += 1
                filtered_messages.append({"role": "system", "content": f"<省略, 共{len(msg.get('content', ''))}字符>"})
            else:
                filtered_messages.append(msg)

        openai_request_log = {
            "model": openai_request.get("model"),
            "messages": filtered_messages,
            "messages_count": len(openai_request.get("messages", [])),
            "system_messages_count": system_count,
            "max_tokens": openai_request.get("max_tokens"),
            "temperature": openai_request.get("temperature"),
            "top_p": openai_request.get("top_p"),
            "stream": openai_request.get("stream"),
            "stop": openai_request.get("stop"),
            "tools_count": len(openai_request.get("tools", [])),
            "tool_choice": openai_request.get("tool_choice"),
        }
        logger.info(f"[Request {request_id}] OpenAI转换请求: {json.dumps(openai_request_log, ensure_ascii=False)}")
        if extra_body:
            logger.info(f"[Request {request_id}] OpenAI extra_body: {json.dumps(extra_body, ensure_ascii=False)}")

        # Check if client disconnected before processing
        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if request.stream:
            # Streaming response - wrap in error handling
            try:
                openai_stream = openai_client.create_chat_completion_stream(
                    openai_request, request_id, extra_body
                )
                return StreamingResponse(
                    convert_openai_streaming_to_claude_with_cancellation(
                        openai_stream,
                        request,
                        logger,
                        http_request,
                        openai_client,
                        request_id,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            except HTTPException as e:
                # Convert to proper error response for streaming
                logger.error(f"Streaming error: {e.detail}")
                import traceback

                logger.error(traceback.format_exc())
                error_message = openai_client.classify_openai_error(e.detail)
                error_response = {
                    "type": "error",
                    "error": {"type": "api_error", "message": error_message},
                }
                return JSONResponse(status_code=e.status_code, content=error_response)
        else:
            # Non-streaming response - pass extra_body
            openai_response = await openai_client.create_chat_completion(
                openai_request, request_id, extra_body
            )
            # 过滤响应中的 content，只保留简短预览
            openai_response_filtered = openai_response.copy()
            if "choices" in openai_response_filtered and openai_response_filtered["choices"]:
                choice = openai_response_filtered["choices"][0].copy()
                if "message" in choice:
                    message = choice["message"].copy()
                    # 只保留 content 的前200字符
                    if "content" in message and message["content"]:
                        content_preview = str(message["content"])[:200] + "..." if len(str(message["content"])) > 200 else message["content"]
                        message["content"] = f"<{content_preview}>"
                    choice["message"] = message
                openai_response_filtered["choices"] = [choice]
            logger.info(f"[Request {request_id}] OpenAI原始响应: {json.dumps(openai_response_filtered, ensure_ascii=False)}")

            claude_response = convert_openai_to_claude_response(
                openai_response, request
            )
            # 过滤 Claude 响应中的 content，只保留简短预览和类型
            claude_response_filtered = claude_response.copy()
            if "content" in claude_response_filtered:
                filtered_content = []
                for block in claude_response_filtered["content"]:
                    if isinstance(block, dict):
                        filtered_block = {"type": block.get("type")}
                        if block.get("type") == "text":
                            text = block.get("text", "")
                            filtered_block["text_preview"] = text[:200] + "..." if len(text) > 200 else text
                            filtered_block["text_length"] = len(text)
                        elif block.get("type") == "thinking":
                            text = block.get("text", "")
                            filtered_block["thinking_length"] = len(text)
                            filtered_block["thinking_preview"] = text[:200] + "..." if len(text) > 200 else text
                        elif block.get("type") == "tool_use":
                            filtered_block["name"] = block.get("name")
                            filtered_block["id"] = block.get("id")
                        filtered_content.append(filtered_block)
                claude_response_filtered["content"] = filtered_content
            logger.info(f"[Request {request_id}] Claude最终响应: {json.dumps(claude_response_filtered, ensure_ascii=False)}")
            return claude_response
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logger.error(f"Unexpected error processing request: {e}")
        logger.error(traceback.format_exc())
        error_message = openai_client.classify_openai_error(str(e))
        raise HTTPException(status_code=500, detail=error_message)


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: ClaudeTokenCountRequest, _: None = Depends(validate_api_key)):
    try:
        # For token counting, we'll use a simple estimation
        # In a real implementation, you might want to use tiktoken or similar

        total_chars = 0

        # Count system message characters
        if request.system:
            if isinstance(request.system, str):
                total_chars += len(request.system)
            elif isinstance(request.system, list):
                for block in request.system:
                    if hasattr(block, "text"):
                        total_chars += len(block.text)

        # Count message characters
        for msg in request.messages:
            if msg.content is None:
                continue
            elif isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text") and block.text is not None:
                        total_chars += len(block.text)

        # Rough estimation: 4 characters per token
        estimated_tokens = max(1, total_chars // 4)

        return {"input_tokens": estimated_tokens}

    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/event_logging/batch")
async def event_logging_batch(request: Request, _: None = Depends(validate_api_key)):
    """批量事件日志记录接口"""
    try:
        try:
            body = await request.json()
        except:
            body = {}

        batch_id = body.get("batch_id") or f"batch_{uuid.uuid4().hex[:16]}"
        events = body.get("events", [])
        if not isinstance(events, list):
            events = [events] if events else []

        processed_count = len(events)

        if events:
            logger.debug(f"Event Logging Batch: batch_id={batch_id}, events_count={processed_count}")
            for i, event in enumerate(events[:5]):
                event_type = event.get("event_type", "unknown") if isinstance(event, dict) else "unknown"
                logger.debug(f"   Event {i+1}: type={event_type}")
            if processed_count > 5:
                logger.debug(f"   ... and {processed_count - 5} more events")

        return EventLoggingBatchResponse(
            success=True,
            batch_id=batch_id,
            processed_count=processed_count,
            message="Events logged successfully"
        )

    except Exception as e:
        logger.warning(f"Event logging batch error: {e}")
        return EventLoggingBatchResponse(
            success=True,
            batch_id=f"batch_{uuid.uuid4().hex[:16]}",
            processed_count=0,
            message="Events received"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_api_configured": bool(config.openai_api_key),
        "api_key_valid": config.validate_api_key(),
        "client_api_key_validation": bool(config.anthropic_api_key),
    }


@router.get("/test-connection")
async def test_connection():
    """Test API connectivity to OpenAI"""
    try:
        # Simple test request to verify API connectivity
        test_response = await openai_client.create_chat_completion(
            {
                "model": config.small_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }
        )

        return {
            "status": "success",
            "message": "Successfully connected to OpenAI API",
            "model_used": config.small_model,
            "timestamp": datetime.now().isoformat(),
            "response_id": test_response.get("id", "unknown"),
        }

    except Exception as e:
        logger.error(f"API connectivity test failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_type": "API Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "suggestions": [
                    "Check your OPENAI_API_KEY is valid",
                    "Verify your API key has the necessary permissions",
                    "Check if you have reached rate limits",
                ],
            },
        )


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Claude-to-OpenAI API Proxy v1.0.0",
        "status": "running",
        "config": {
            "openai_base_url": config.openai_base_url,
            "max_tokens_limit": config.max_tokens_limit,
            "api_key_configured": bool(config.openai_api_key),
            "client_api_key_validation": bool(config.anthropic_api_key),
            "big_model": config.big_model,
            "small_model": config.small_model,
        },
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection",
        },
    }
