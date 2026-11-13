"""MCP server over stdio: the web UI's Slice and Stitch plan as tools for an AI client.

    python3 digitize.py mcp              tools only, on stdin/stdout
    python3 digitize.py mcp --port 8765  also serve the page, so the browser shows each run

The tools drive the same Flask app as the page (server.create_app), in process through
its test client, so a tool call and a button press run the same job code, the same checks
and the same one-job-at-a-time lock. Nothing here starts a pipeline step on its own.

The protocol is the small part of MCP the tools need (initialize, tools/list, tools/call,
ping), as newline-delimited JSON-RPC 2.0 on stdin/stdout, written with the standard library
so the tool keeps its no-new-dependency rule (docs/design/webui.md D8). stdout carries only
protocol messages; everything else goes to stderr.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from tools.digitize.web import server

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INFO = {"name": "digitize-patches", "version": "0.1.0"}
POLL_S = 0.25
SLICE_WAIT_S = 120             # a Slice takes 3 to 5 s on a patch-sized picture
PLAN_WAIT_S = 60               # an export takes 24 s to 85 min cold; status picks up the rest

OVERRIDES_SCHEMA = {
    "type": "array",
    "description": "Stitch type overrides, each a point in source picture pixels (x right, "
                   "y down) and the stitch type for the Region holding it. Replaces the list "
                   "the last Slice used; [] clears it. Left out: keep the last list.",
    "items": {
        "type": "object",
        "properties": {
            "at_px": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "type": {"type": "string", "enum": ["satin", "fill"]},
        },
        "required": ["at_px", "type"],
    },
}

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "slice",
        "title": "Slice a picture",
        "description": (
            "Turn a raster picture into the Region map: key-background and vectorize, about "
            "3 to 5 s, no stitches yet. Returns the result of every stitch type override "
            "(applied, refused with its coverage %, orphaned, shadowed) and the files made. "
            "Settings left out keep their value from the last Slice in this session."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "picture": {"type": "string", "description":
                            "Path to a png, jpg, jpeg or webp file. Needed on the first call; "
                            "a new picture clears the overrides."},
                "hoop": {"type": "array", "items": {"type": "number"}, "minItems": 2,
                         "maxItems": 2, "description": "Hoop in mm, [120, 120] or [260, 200]."},
                "size_mm": {"type": ["number", "null"], "description":
                            "Patch size in mm; null fills the hoop. Default 100."},
                "colors": {"type": "integer", "enum": [4, 6, 8]},
                "key_background": {"type": "boolean",
                                   "description": "Remove a plain background first."},
                "config": {"type": "string", "description":
                           "Path to a YAML config (palette, profile, satinize, ...)."},
                "yaml_text": {"type": "string", "description": "YAML config as text."},
                "overrides": OVERRIDES_SCHEMA,
            },
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    },
    {
        "name": "make_stitch_plan",
        "title": "Make the stitch plan",
        "description": (
            "Export the last Slice's SVG to VP3 through Ink/Stitch and render the Stitch plan. "
            "Takes about 10 s on a warm cache and up to many minutes cold. Waits up to wait_s "
            "seconds; when the export is still running, call status later."),
        "inputSchema": {
            "type": "object",
            "properties": {"wait_s": {"type": "number", "minimum": 0,
                                      "description": "Seconds to wait, default %d." % PLAN_WAIT_S}},
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False},
    },
    {
        "name": "status",
        "title": "Job status",
        "description": (
            "The current job: state (idle, slicing, exporting, failed, cancelled), stage, "
            "elapsed time, the last Slice with its override results, and the Stitch plan "
            "(stitches, size, threads, warnings) with its picture when one exists."),
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "cancel",
        "title": "Cancel the running job",
        "description": "Stop the running Slice or export and every process it started.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "openWorldHint": False},
    },
]


class ToolError(Exception):
    """A tool failed in a way the client should read: plain words, isError: true."""


class Tools:
    """The four tools, on one app. Settings from the last Slice carry to the next one."""

    def __init__(self, app, port: Optional[int] = None):
        self.app = app
        self.client = app.test_client()
        # With --port the app checks Host against that port, so speak as its own page.
        self.base_url = "http://127.0.0.1:%d" % port if port else "http://127.0.0.1"
        self.out_root = Path(app.job.out_root)
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
        """/files/<slug>/<name>?v=3 -> the file on disk, or None."""
        if not url:
            return None
        slug, name = url.split("?", 1)[0].split("/")[2:4]
        path = self.out_root / slug / name
        return path if path.is_file() else None

    # -- the tools ----------------------------------------------------------------------

    def slice(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
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
        wanted = dict(self.defaults)
        wanted.update(self.settings)
        for key in self.defaults:
            if key in args:
                wanted[key] = args[key]
        body = {k: wanted[k] for k in self.defaults}
        self._call("POST", "/api/slice", json=body)
        self.settings = body
        status = self._wait(SLICE_WAIT_S)
        return self._report(status, "slice")

    def make_stitch_plan(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._call("POST", "/api/stitchplan")
        wait_s = args.get("wait_s")
        status = self._wait(PLAN_WAIT_S if wait_s is None else float(wait_s))
        return self._report(status, "stitchplan")

    def status(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        return self._report(self._call("GET", "/api/status"), "status")

    def cancel(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [text(self._call("POST", "/api/cancel"))]

    # -- what a call returns ------------------------------------------------------------

    def _report(self, status: Dict[str, Any], asked: str) -> List[Dict[str, Any]]:
        """The status, with file paths in place of page URLs, override results and pictures."""
        out: Dict[str, Any] = {k: status.get(k) for k in ("state", "stage", "elapsed_s", "error")}
        if status["state"] == "failed":
            out["log_tail"] = status.get("log_tail")
        if status.get("picture"):
            out["picture"] = {"path": str(self._file(status["picture"]["url"]) or ""),
                              "width": status["picture"]["width"],
                              "height": status["picture"]["height"]}
        images: List[Path] = []
        sl = status.get("slice")
        if sl:
            regions = self._file(sl.get("regions_url"))
            out["slice"] = {
                "slice_id": sl["slice_id"], "hoop": sl["hoop"],
                "svg": str(self._file(sl["svg_url"]) or ""),
                "regions": str(regions or ""),
                "overrides": self._override_results(sl, regions),
            }
            if asked == "slice" and self._file(sl.get("posterized_url")):
                images.append(self._file(sl["posterized_url"]))
        plan = status.get("stitchplan")
        if plan:
            out["stitchplan"] = dict(
                {k: plan.get(k) for k in ("slice_id", "stitches", "width_mm", "height_mm",
                                          "threads", "warnings", "removed_by_precheck")},
                stale=bool(sl) and plan["slice_id"] != sl["slice_id"],       # E7
                vp3=str(self._file(plan["vp3_url"]) or ""),
                stitchplan_png=str(self._file(plan["stitchplan_url"]) or ""),
                coverage_png=str(self._file(plan["coverage_url"]) or ""))
            if asked != "slice" and self._file(plan["stitchplan_url"]):
                images.append(self._file(plan["stitchplan_url"]))
        if status["state"] in server.RUNNING_STATES:
            out["note"] = "Still running. Call status again later."
        content = [text(out)]
        for path in images:
            content.append({"type": "image", "mimeType": "image/png",
                            "data": base64.b64encode(path.read_bytes()).decode("ascii")})
        return content

    @staticmethod
    def _override_results(sl: Dict[str, Any], regions: Optional[Path]) -> List[Dict[str, Any]]:
        """What the pipeline did with each override (regions.json), else what was sent."""
        if regions:
            try:
                reported = json.loads(regions.read_text(encoding="utf-8")).get("overrides")
                if isinstance(reported, list):
                    return reported
            except (OSError, ValueError):
                pass
        return [dict(o, status="pending") for o in sl.get("overrides") or []]


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
        self.handlers: Dict[str, Callable[[Dict[str, Any]], List[Dict[str, Any]]]] = {
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
            content = handler(params.get("arguments") or {})
            self.reply(msg_id, {"content": content, "isError": False})
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
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
                "instructions": "Slice a picture to see its Regions and override results, "
                                "then make_stitch_plan for the VP3 and its Stitch plan.",
            })
        elif method == "ping":
            self.reply(msg_id, {})
        elif method == "tools/list":
            self.reply(msg_id, {"tools": TOOLS})
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
        import flask.cli
        flask.cli.show_server_banner = lambda *a, **k: None    # it prints to stdout
        threading.Thread(target=app.run, name="digitize-web-page", daemon=True,
                         kwargs={"host": "127.0.0.1", "port": ns.port, "threaded": True,
                                 "debug": False, "use_reloader": False}).start()
        print("Digitize Patches page: http://127.0.0.1:%d/" % ns.port, file=sys.stderr)
    sys.stderr.flush()
    # Flask's startup banner and request log go to stderr; stdout stays protocol only.
    StdioServer(Tools(app, ns.port)).serve(sys.stdin)
    return 0


if __name__ == "__main__":
    sys.exit(main())
