"""Zero-extra-dependency HTTP API for occupation Top10 + LLM analysis.

Run:
    python -m src.llm.occupation_agent_api --host 127.0.0.1 --port 8120

Endpoint:
    POST /occupation/analyze
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
from typing import Any

from src.llm.occupation_agent_service import (
    OccupationAgentService,
    dumps_response,
    parse_analysis_request,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class OccupationAgentHandler(BaseHTTPRequestHandler):
    """HTTP handler exposing the occupation analysis service."""

    service: OccupationAgentService | None = None

    def do_GET(self) -> None:  # noqa: N802
        """Expose a lightweight health check."""
        if self.path != "/health":
            self._write_json({"error": "not_found", "message": "未知路径"}, HTTPStatus.NOT_FOUND)
            return
        self._write_json({"status": "ok", "service": "occupation-agent-api"})

    def do_POST(self) -> None:  # noqa: N802
        """Handle occupation analysis requests."""
        if self.path != "/occupation/analyze":
            self._write_json({"error": "not_found", "message": "未知路径"}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json_body()
            request = parse_analysis_request(payload)
            if self.service is None:
                raise RuntimeError("OccupationAgentService 尚未初始化")
            result = self.service.analyze(request)
            self._write_json(result)
        except ValueError as exc:
            self._write_json({"error": "bad_request", "message": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            logger.exception("职业细类 Agent API 调用失败")
            self._write_json(
                {"error": "internal_error", "message": str(exc)},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            raise ValueError("请求体不能为空")
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"请求体不是合法 JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON object")
        return payload

    def _write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = dumps_response(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Route BaseHTTPRequestHandler logs through logging."""
        logger.info("%s - %s", self.address_string(), format % args)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the API server."""
    parser = argparse.ArgumentParser(description="职业细类 Agent API: BGE Top10 + llama/vLLM 报告")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8120, help="监听端口")
    parser.add_argument(
        "--model-path",
        default=None,
        help="覆盖 fine-tuned embedding 模型路径，默认读取 config/database.yaml",
    )
    parser.add_argument(
        "--force-rebuild-index",
        action="store_true",
        help="强制重建职业词典 embedding 缓存",
    )
    return parser


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8120,
    model_path: str | Path | None = None,
    force_rebuild_index: bool = False,
) -> None:
    """Initialize the service and block serving HTTP requests."""
    logger.info("初始化职业细类 Agent 服务，首次加载模型可能需要一些时间...")
    OccupationAgentHandler.service = OccupationAgentService(
        model_path=model_path,
        force_rebuild_index=force_rebuild_index,
    )
    server = ThreadingHTTPServer((host, port), OccupationAgentHandler)
    logger.info("职业细类 Agent API 已启动: http://%s:%s", host, port)
    logger.info("健康检查: GET /health")
    logger.info("分析接口: POST /occupation/analyze")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭服务。")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args(argv)
    run_server(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        force_rebuild_index=args.force_rebuild_index,
    )


if __name__ == "__main__":
    main()
