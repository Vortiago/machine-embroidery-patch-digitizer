"""MCP server over stdio: the web UI's Slice and Stitch plan as tools for an AI client.

slice, make_stitch_plan and status also name an MCP App view (static/mcp_app.html), so a
client that supports MCP Apps draws the Region map and the Stitch plan in the chat.

    python3 digitize.py mcp              tools only, on stdin/stdout
    python3 digitize.py mcp --port 8765  also serve the page, so the browser shows each run

The tools drive the same Flask app as the page (server.create_app), in process through
its test client, so a tool call and a button press run the same job code, the same checks
and the same one-job-at-a-time lock. Nothing here starts a pipeline step on its own.

The protocol is the small part of MCP the tools need (initialize, ping, tools/list,
tools/call, and resources/list and resources/read for the view), as newline-delimited
JSON-RPC 2.0 on stdin/stdout, written with the standard library so the tool keeps its
no-new-dependency rule (docs/design/webui.md D8). stdout carries only protocol messages;
everything else goes to stderr.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from werkzeug.serving import make_server

from tools.digitize.web import server

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "digitize-patches", "version": "0.1.0"}
POLL_S = 0.25
SLICE_WAIT_S = 120             # a Slice takes 3 to 5 s on a patch-sized picture
PLAN_WAIT_S = 60               # an export takes 24 s to 85 min cold; status picks up the rest
VIEW_URI = "ui://digitize/view.html"
VIEW_MIME = "text/html;profile=mcp-app"
VIEW_FILE = Path(__file__).resolve().parent / "static" / "mcp_app.html"
VIEW_SVG_MAX = 2_000_000       # bytes of Region map SVG sent to the view; larger is left out

TOOLS_FILE = Path(__file__).resolve().parent / "static" / "mcp_tools.json"


def load_tools() -> List[Dict[str, Any]]:
    """The tool list shared with the page's WebMCP tools (static/mcp_tools.json), plus what
    only the stdio server has: picture and config paths, and the MCP App view."""
    tools = json.loads(TOOLS_FILE.read_text(encoding="utf-8"))
    for tool in tools:
        if tool["name"] == "slice":
            tool["inputSchema"]["properties"].update({
                "picture": {"type": "string", "description":
                            "Path to a png, jpg, jpeg or webp file. Needed on the first call; "
                            "a new picture clears the overrides."},
                "config": {"type": "string", "description":
                           "Path to a YAML config (palette, profile, satinize, ...)."},
                "yaml_text": {"type": "string", "description": "YAML config as text."},
            })
        if tool["name"] != "cancel":
            tool["_meta"] = {"ui": {"resourceUri": VIEW_URI}}
    return tools


TOOLS = load_tools()
Report = Tuple[List[Dict[str, Any]], Dict[str, Any]]      # (content, view for the MCP App)


class ToolError(Exception):
    """A tool failed in a way the client should read: plain words, isError: true."""


class Tools:
    """The four tools, on one app. Settings from the last Slice carry to the next one."""

    def __init__(self, app, port: Optional[int] = None):
        self.app = app
        self.client = app.test_client()
        # With --port the app checks Host against that port, so speak as its own page.
        self.base_url = "http://127.0.0.1:%d" % port if port else "http://127.0.0.1"
        self.settings: Dict[str, Any] = {}
        self.picture: Optional[Path] = None
        options = self._call("GET", "/api/options")
        self.defaults = {"hoop": options["hoops"][options["default_hoop"]],
                         "size_mm": options["default_size_mm"],
                         "colors": options["default_colors"],
                         "key_background": False, "yaml_text": "", "overrides": []}

    # -- talking to the app ---------------------------------------------------------

    def _call(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        res = self.client.open(path, method=method, base_url=self.base_url, **kwargs)
        data = res.get_json(silent=True) or {}
        if res.status_code == 409:
            raise ToolError("A job is already running. Call status, or cancel it first.")
        if res.status_code >= 400:
            raise ToolError(data.get("error") or "%s %s failed (%d)" % (method, path, res.status_code))
        return data

    def _wait(self, wait_s: float) -> Dict[str, Any]:
        deadline = time.time() + max(0.0, wait_s)
        while True:
            status = self._call("GET", "/api/status")
            if status["state"] not in server.RUNNING_STATES or time.time() >= deadline:
                return status
            time.sleep(POLL_S)

    def _file(self, url: Optional[str]) -> Optional[Path]:
        """A /files/<slug>/<name>?v=3 URL of the current job -> that file on disk, or None."""
        jobdir = self.app.job.dir
        if not url or jobdir is None:
            return None
        path = jobdir / url.split("?", 1)[0].rsplit("/", 1)[-1]
        return path if path.is_file() else None

    # -- the tools ----------------------------------------------------------------------

    def slice(self, args: Dict[str, Any]) -> Report:
        picture = args.get("picture")
        if picture:
            path = Path(picture).expanduser().resolve()
            if not path.is_file():
                raise ToolError("There is no picture file at %s." % path)
            if path != self.picture or self.app.job.image is None:
                with open(str(path), "rb") as handle:
                    self._call("POST", "/api/picture", content_type="multipart/form-data",
                               data={"picture": (handle, path.name)})
                self.picture = path
                self.settings.pop("overrides", None)          # E5
        elif self.app.job.image is None:
            raise ToolError("No picture yet: pass picture, a path to a png, jpg or webp file.")
        if args.get("config"):
            config = Path(args["config"]).expanduser()
            try:
                args = dict(args, yaml_text=config.read_text(encoding="utf-8"))
            except OSError as exc:
                raise ToolError("The config %s could not be read (%s)." % (config, exc))
        body = {k: args.get(k, self.settings.get(k, v)) for k, v in self.defaults.items()}
        self._call("POST", "/api/slice", json=body)
        self.settings = body
        return self._report(self._wait(SLICE_WAIT_S), after_slice=True)

    def make_stitch_plan(self, args: Dict[str, Any]) -> Report:
        self._call("POST", "/api/stitchplan")
        wait_s = args.get("wait_s")
        return self._report(self._wait(PLAN_WAIT_S if wait_s is None else float(wait_s)))

    def status(self, args: Dict[str, Any]) -> Report:
        return self._report(self._call("GET", "/api/status"))

    def cancel(self, args: Dict[str, Any]) -> Report:
        return [text(self._call("POST", "/api/cancel"))], {}

    # -- what a call returns ------------------------------------------------------------

    def _report(self, status: Dict[str, Any], after_slice: bool = False) -> Report:
        """The status, with file paths in place of page URLs, override results and pictures.

        Returns (content, view): view goes in the result's _meta for the MCP App, which
        draws the Region map from the SVG and object kinds and cannot fetch files itself.
        After a Slice the picture is the posterized preview, otherwise the Stitch plan.
        """
        view: Dict[str, Any] = {}
        out: Dict[str, Any] = {k: status.get(k) for k in ("state", "stage", "elapsed_s", "error")}
        if status["state"] == "failed":
            out["log_tail"] = status.get("log_tail")
        if status.get("picture"):
            out["picture"] = {"path": str(self._file(status["picture"]["url"]) or ""),
                              "width": status["picture"]["width"],
                              "height": status["picture"]["height"]}
        image: Optional[Path] = None
        sl = status.get("slice")
        if sl:
            svg, regions_file = self._file(sl["svg_url"]), self._file(sl.get("regions_url"))
            regions = read_json(regions_file)
            reported = regions.get("overrides")
            out["slice"] = {
                "slice_id": sl["slice_id"], "hoop": sl["hoop"],
                "svg": str(svg or ""), "regions": str(regions_file or ""),
                # what the pipeline did with each override, else what was sent
                "overrides": reported if isinstance(reported, list) else
                [dict(o, status="pending") for o in sl.get("overrides") or []],
            }
            if svg and svg.stat().st_size <= VIEW_SVG_MAX:
                view["svg"] = svg.read_text(encoding="utf-8", errors="replace")
            objects = regions.get("objects")
            if isinstance(objects, list):
                view["kinds"] = {o["id"]: o["kind"] for o in objects
                                 if isinstance(o, dict) and "id" in o and "kind" in o}
            if after_slice:
                image = self._file(sl.get("posterized_url"))
        plan = status.get("stitchplan")
        if plan:
            plan_png = self._file(plan["stitchplan_url"])
            out["stitchplan"] = dict(
                {k: plan.get(k) for k in ("slice_id", "stitches", "width_mm", "height_mm",
                                          "threads", "warnings", "removed_by_precheck")},
                stale=bool(sl) and plan["slice_id"] != sl["slice_id"],       # E7
                vp3=str(self._file(plan["vp3_url"]) or ""),
                stitchplan_png=str(plan_png or ""),
                coverage_png=str(self._file(plan["coverage_url"]) or ""))
            if not after_slice and plan_png:
                image = plan_png
                view["plan_image"] = True
        if status["state"] in server.RUNNING_STATES:
            out["note"] = "Still running. Call status again later."
        content = [text(out)]
        if image:
            content.append({"type": "image", "mimeType": "image/png",
                            "data": base64.b64encode(image.read_bytes()).decode("ascii")})
        return content, view


def read_json(path: Optional[Path]) -> Dict[str, Any]:
    """A JSON object from a file, or {} when there is none or it does not parse."""
    if not path:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def text(value: Any) -> Dict[str, Any]:
    return {"type": "text", "text": json.dumps(value, indent=1)}


# ---- JSON-RPC over stdio ------------------------------------------------------------------


class StdioServer:
    """Newline-delimited JSON-RPC 2.0. Each tools/call runs in its own thread, so cancel
    and status answer while make_stitch_plan waits; the job lock keeps it one job."""

    def __init__(self, tools: Tools, out=None):
        self.tools = tools
        self.out = out or sys.stdout
        self.write_lock = threading.Lock()
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            t["name"]: getattr(tools, t["name"]) for t in TOOLS}

    def send(self, message: Dict[str, Any]) -> None:
        with self.write_lock:
            self.out.write(json.dumps(message, separators=(",", ":")) + "\n")
            self.out.flush()

    def reply(self, msg_id: Any, result: Any = None, error: Optional[Dict[str, Any]] = None):
        message: Dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}
        if error is not None:
            message["error"] = error
        else:
            message["result"] = result
        self.send(message)

    def call_tool(self, msg_id: Any, params: Dict[str, Any]) -> None:
        name = params.get("name")
        handler = self.handlers.get(name)
        if handler is None:
            self.reply(msg_id, error={"code": -32602, "message": "Unknown tool: %s" % name})
            return
        try:
            content, view = handler(params.get("arguments") or {})
            result: Dict[str, Any] = {"content": content, "isError": False}
            if view:
                result["_meta"] = {"digitize/view": view}
            self.reply(msg_id, result)
        except ToolError as exc:
            self.reply(msg_id, {"content": [{"type": "text", "text": str(exc)}], "isError": True})
        except Exception as exc:                         # a bug: say so, keep the server up
            self.reply(msg_id, {"content": [{"type": "text", "text": "Internal error: %r" % exc}],
                                "isError": True})

    def handle(self, message: Any) -> Optional[threading.Thread]:
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            self.reply(None, error={"code": -32600, "message": "Invalid request"})
            return None
        method, msg_id = message.get("method"), message.get("id")
        params = message.get("params") or {}
        if "id" not in message:                          # a notification: nothing to answer
            return None
        if method == "initialize":
            asked = params.get("protocolVersion")
            self.reply(msg_id, {
                "protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                "capabilities": {"tools": {}, "resources": {},
                                 "extensions": {"io.modelcontextprotocol/ui": {}}},
                "serverInfo": SERVER_INFO,
                "instructions": "Slice a picture to see its Regions and override results, "
                                "then make_stitch_plan for the VP3 and its Stitch plan.",
            })
        elif method == "ping":
            self.reply(msg_id, {})
        elif method == "tools/list":
            self.reply(msg_id, {"tools": TOOLS})
        elif method == "resources/list":
            self.reply(msg_id, {"resources": [{"uri": VIEW_URI, "name": "Region map and Stitch plan",
                                               "mimeType": VIEW_MIME}]})
        elif method == "resources/read":
            if params.get("uri") != VIEW_URI:
                self.reply(msg_id, error={"code": -32002, "message": "Resource not found: %s"
                                          % params.get("uri")})
            else:
                self.reply(msg_id, {"contents": [{
                    "uri": VIEW_URI, "mimeType": VIEW_MIME,
                    "text": VIEW_FILE.read_text(encoding="utf-8"),
                    "_meta": {"ui": {"prefersBorder": True}}}]})
        elif method == "tools/call":
            worker = threading.Thread(target=self.call_tool, args=(msg_id, params), daemon=True)
            worker.start()
            return worker
        else:
            self.reply(msg_id, error={"code": -32601, "message": "Method not found: %s" % method})
        return None

    def serve(self, lines) -> None:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                self.reply(None, error={"code": -32700, "message": "Parse error"})
                continue
            self.handle(message)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="digitize.py mcp",
                                 description="MCP server on stdio for Slice and Stitch plan.")
    ap.add_argument("--port", type=int, default=None,
                    help="also serve the web page on 127.0.0.1:PORT, sharing the same job")
    ns = ap.parse_args(args)
    app = server.create_app(server.REPO_ROOT, port=ns.port)
    if ns.port:
        # make_server prints no banner, so stdout stays protocol only; request logs go to stderr.
        page = make_server("127.0.0.1", ns.port, app, threaded=True)
        threading.Thread(target=page.serve_forever, name="digitize-web-page", daemon=True).start()
        print("Digitize Patches page: http://127.0.0.1:%d/" % ns.port, file=sys.stderr)
        sys.stderr.flush()
    StdioServer(Tools(app, ns.port)).serve(sys.stdin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
