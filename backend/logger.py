# 日志记录模块
from backend.database import get_agent_logs as db_get_logs
from backend.database import save_agent_log


def log_agent_call(
    conversation_id: str,
    question: str,
    steps: list[dict],
    final_answer: str,
    model: str,
    total_tokens: int,
    elapsed_ms: float,
    success: bool = True,
    error: str = "",
) -> int:
    """记录 Agent 调用的完整日志"""
    # 提取 reasoning 摘要
    reasoning_parts = []
    tool_calls = []
    
    for step in steps:
        step_type = step.get("step_type", "")
        if step_type == "thought":
            content = step.get("content", "")
            if content:
                reasoning_parts.append(content[:200])
        elif step_type == "action":
            tool_calls.append({
                "tool": step.get("tool_name", ""),
                "args": step.get("tool_args", {}),
            })
    
    log_data = {
        "conversation_id": conversation_id,
        "question": question,
        "reasoning": " | ".join(reasoning_parts),
        "tool_calls": tool_calls,
        "final_answer": final_answer,
        "model": model,
        "total_tokens": total_tokens,
        "elapsed_ms": round(elapsed_ms, 2),
        "success": success,
        "error": error,
    }
    
    return save_agent_log(log_data)


def get_logs(limit: int = 50) -> list[dict]:
    """获取日志列表"""
    return db_get_logs(limit)
