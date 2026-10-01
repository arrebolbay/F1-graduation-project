#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
F1 摩纳哥站进站策略优化系统 — Web 服务器(纯标准库,零第三方依赖)

启动:
    python server.py            # 仅启动服务
    python server.py --open     # 启动并自动打开浏览器
    python server.py --port 9000

路由:
    GET  /                      前端页面
    GET  /static/*              静态资源(html/css/js)
    GET  /charts/*              图表 PNG(图表\ 目录)
    GET  /api/meta              车手/配方/场景/默认参数
    GET  /api/prerace           赛前 DP 策略(?driver=LEC&mode=normal|two_stop)
    GET  /api/history           历史记录
    POST /api/simulate          赛中 Stackelberg 蒙特卡洛(JSON 请求体)
    POST /api/history/clear     清空历史记录
"""

import argparse
import json
import mimetypes
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import engine

WEB_DIR = engine.WEB_DIR
STATIC_DIR = os.path.join(WEB_DIR, "static")
CHART_DIR = engine.CHART_DIR


def _safe_join(base, *parts):
    """拼接路径并阻断目录穿越。"""
    path = os.path.realpath(os.path.join(base, *parts))
    base_real = os.path.realpath(base)
    if path != base_real and not path.startswith(base_real + os.sep):
        raise ValueError("非法路径")
    return path


class Handler(BaseHTTPRequestHandler):
    server_version = "F1PitWall/1.0"

    # ---------------------------------------------------------- 基础响应
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path):
        if not os.path.isfile(path):
            self._json({"error": "文件不存在"}, 404)
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=300")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        charset = "utf-8"
        ctype = (self.headers.get("Content-Type") or "").lower()
        if "charset=" in ctype:
            charset = ctype.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
        last_err = None
        for enc in dict.fromkeys([charset, "utf-8", "gbk"]):
            try:
                return json.loads(raw.decode(enc) or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                last_err = e
                continue
        raise json.JSONDecodeError(f"无法解码请求体: {last_err}", "", 0)

    # ---------------------------------------------------------- GET 路由
    def do_GET(self):
        url = urlparse(self.path)
        path = unquote(url.path)
        try:
            if path in ("/", "/index.html"):
                self._file(os.path.join(STATIC_DIR, "index.html"))
            elif path.startswith("/static/"):
                rel = path[len("/static/"):].strip("/").split("/")
                self._file(_safe_join(STATIC_DIR, *rel))
            elif path.startswith("/charts/"):
                rel = path[len("/charts/"):].strip("/").split("/")
                self._file(_safe_join(CHART_DIR, *rel))
            elif path == "/api/meta":
                self._json(engine.get_meta())
            elif path == "/api/prerace":
                q = parse_qs(url.query)
                driver = (q.get("driver") or [""])[0]
                mode = (q.get("mode") or ["normal"])[0]
                log = (q.get("log") or ["1"])[0] != "0"
                result = engine.get_prerace(driver, mode)
                if log:
                    engine.append_history({
                    "type": "prerace",
                    "title": f"赛前模拟 · {result['name']}({driver})",
                    "detail": f"模式: {result['mode_name']} | {result['strategy_text']}",
                    "result": (f"预计总用时 {result['total_display']} | "
                               f"进站 {result['stops']} 次"
                               + (f"(第 {','.join(map(str, result['pit_laps']))} 圈)"
                                  if result["pit_laps"] else "")),
                    "payload": {"type": "prerace", "driver": driver, "mode": mode},
                })
                self._json(result)
            elif path == "/api/tables":
                self._json(engine.get_tables())
            elif path == "/api/history":
                q = parse_qs(url.query)
                limit = int((q.get("limit") or ["200"])[0])
                self._json({"records": engine.read_history(limit)})
            else:
                self._json({"error": f"未知路径: {path}"}, 404)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"服务器内部错误: {e}"}, 500)

    # ---------------------------------------------------------- POST 路由
    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/prerace":
                req = self._read_body()
                result = engine.get_prerace(
                    req.get("driver"), req.get("mode", "normal"),
                    tires=req.get("tires"),
                )
                if req.get("log", True):
                    engine.append_history({
                        "type": "prerace",
                        "title": f"赛前模拟 · {result['name']}({result['driver']})",
                        "detail": f"模式: {result['mode_name']} | {result['strategy_text']}",
                        "result": (f"预计总用时 {result['total_display']} | "
                                   f"进站 {result['stops']} 次"
                                   + (f"(第 {','.join(map(str, result['pit_laps']))} 圈)"
                                      if result["pit_laps"] else "")),
                        "payload": {"type": "prerace", "driver": result["driver"],
                                    "mode": result["mode"],
                                    "tires": result.get("tires")},
                    })
                self._json(result)
            elif path == "/api/simulate":
                req = self._read_body()
                result = engine.run_simulation(
                    scenario=req.get("scenario"),
                    my_driver=req.get("my_driver"),
                    my_compound=req.get("my_compound"),
                    my_age=req.get("my_age"),
                    rival_driver=req.get("rival_driver"),
                    rival_compound=req.get("rival_compound"),
                    rival_age=req.get("rival_age"),
                    current_lap=req.get("current_lap"),
                    gap_s=req.get("gap_s"),
                    sc_type=req.get("sc_type", "SC"),
                    n_sim=req.get("n_sim"),
                    params=req.get("params"),
                    seed=req.get("seed"),
                    my_fit_compound=req.get("my_fit_compound"),
                    my_fit_wear=req.get("my_fit_wear", 0),
                    rival_fit_compound=req.get("rival_fit_compound"),
                    rival_fit_wear=req.get("rival_fit_wear", 0),
                    tires=req.get("tires"),
                )
                engine.append_history({
                    "type": "simulate",
                    "title": f"赛中决策 · {result['scenario_name']}",
                    "detail": (f"{result['inputs']['my']['name']}({result['inputs']['my']['compound']},"
                               f"胎龄{result['inputs']['my']['age']}) vs "
                               f"{result['inputs']['rival']['name']}({result['inputs']['rival']['compound']},"
                               f"胎龄{result['inputs']['rival']['age']}) | "
                               f"第{result['current_lap']}圈 | 差距{result['gap_s']}s"),
                    "result": result["recommendation"]["text"],
                    "payload": {
                        "type": "simulate", "scenario": result["scenario"],
                        "sc_type": result["sc_type"],
                        "my_driver": result["inputs"]["my"]["driver"],
                        "my_compound": result["inputs"]["my"]["compound"],
                        "my_age": result["inputs"]["my"]["age"],
                        "rival_driver": result["inputs"]["rival"]["driver"],
                        "rival_compound": result["inputs"]["rival"]["compound"],
                        "rival_age": result["inputs"]["rival"]["age"],
                        "current_lap": result["current_lap"],
                        "gap_s": result["gap_s"], "n_sim": result["n_sim"],
                        "my_fit_compound": result["inputs"]["my"]["fit_compound"],
                        "my_fit_wear": result["inputs"]["my"]["fit_wear"],
                        "rival_fit_compound": result["inputs"]["rival"]["fit_compound"],
                        "rival_fit_wear": result["inputs"]["rival"]["fit_wear"],
                        "tires": req.get("tires"),
                    },
                })
                self._json(result)
            elif path == "/api/ai":
                req = self._read_body()
                self._json(engine.ai_analysis(req.get("result")))
            elif path == "/api/history/clear":
                engine.clear_history()
                self._json({"ok": True})
            else:
                self._json({"error": f"未知路径: {path}"}, 404)
        except json.JSONDecodeError:
            self._json({"error": "请求体不是合法 JSON"}, 400)
        except ValueError as e:
            self._json({"error": str(e)}, 400)
        except Exception as e:  # noqa: BLE001
            self._json({"error": f"服务器内部错误: {e}"}, 500)

    def log_message(self, fmt, *args):
        sys.stderr.write("[web] %s %s\n" % (self.address_string(), fmt % args))

def main():
    ap = argparse.ArgumentParser(description="F1 摩纳哥停站策略系统 Web 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()

    # 端口自动回退: 8765 被其它程序(如输入法)占用时,自动尝试后续端口
    srv = None
    port = args.port
    for _ in range(10):
        try:
            srv = ThreadingHTTPServer((args.host, port), Handler)
            break
        except OSError:
            print(f" [提示] 端口 {port} 被占用,尝试 {port + 1} ...")
            port += 1
    if srv is None:
        print(" [错误] 无可用端口,请关闭占用程序后重试。")
        return
    args.port = port
    url = f"http://{args.host}:{args.port}/"
    print("=" * 56)
    print(" F1 摩纳哥站进站策略优化系统 · Web 前端")
    print(f" 地址: {url}")
    print(f" 页面: 赛前模拟 / 赛中Stackelberg决策 / 数据总览 / 历史记录")
    print(" 按 Ctrl+C 停止服务")
    print("=" * 56)

    if args.open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()