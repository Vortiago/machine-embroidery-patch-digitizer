"""Tests for the stdio MCP server: the protocol, and each tool against the stand-in scripts.

The app, the stand-in run.sh and export_svg.sh and the picture come from test_web_server,
so a tool call runs the same job code as a button press on the page.

    python3 -m unittest tools.digitize.tests.test_mcp_server -v
"""
from __future__ import annotations

import base64
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.digitize.tests.test_web_server import ServerTestCase, png_bytes  # noqa: E402
from tools.digitize.web import mcp_server  # noqa: E402


class McpTestCase(ServerTestCase):
    def setUp(self):
        super().setUp()
        self.tools = mcp_server.Tools(self.app)
        self.out = io.StringIO()
        self.rpc = mcp_server.StdioServer(self.tools, out=self.out)
        self.picture = self.root / "Skyfowl Patch.png"
        self.picture.write_bytes(png_bytes())

    def request(self, method, params=None, msg_id=1):
        """One request through the stdio loop; returns its reply."""
        self.out.seek(0)
        self.out.truncate()
        worker = self.rpc.handle({"jsonrpc": "2.0", "id": msg_id, "method": method,
                                  "params": params or {}})
        if worker is not None:
            worker.join(30)
        lines = [json.loads(ln) for ln in self.out.getvalue().splitlines() if ln]
        self.assertEqual(len(lines), 1, lines)
        return lines[0]

    def call(self, name, **arguments):
        reply = self.request("tools/call", {"name": name, "arguments": arguments})
        result = reply["result"]
        return result, (json.loads(result["content"][0]["text"])
                        if not result["isError"] else result["content"][0]["text"])


class Protocol(McpTestCase):
    def test_initialize_echoes_a_known_version(self):
        reply = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                            "clientInfo": {"name": "t", "version": "0"}})
        self.assertEqual(reply["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("tools", reply["result"]["capabilities"])
        unknown = self.request("initialize", {"protocolVersion": "1999-01-01"})
        self.assertEqual(unknown["result"]["protocolVersion"], mcp_server.PROTOCOL_VERSIONS[0])

    def test_tools_list(self):
        names = [t["name"] for t in self.request("tools/list")["result"]["tools"]]
        self.assertEqual(names, ["slice", "make_stitch_plan", "status", "cancel"])

    def test_notification_gets_no_reply(self):
        self.assertIsNone(self.rpc.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertEqual(self.out.getvalue(), "")

    def test_unknown_method_and_tool(self):
        self.assertEqual(self.request("nope")["error"]["code"], -32601)
        self.assertEqual(self.request("tools/call", {"name": "nope"})["error"]["code"], -32602)

    def test_serve_reads_lines_and_skips_bad_json(self):
        self.rpc.serve(['{"jsonrpc":"2.0","id":7,"method":"ping"}\n', "\n", "not json\n"])
        lines = [json.loads(ln) for ln in self.out.getvalue().splitlines()]
        self.assertEqual(lines[0], {"jsonrpc": "2.0", "id": 7, "result": {}})
        self.assertEqual(lines[1]["error"]["code"], -32700)


class AppView(McpTestCase):
    def test_tools_name_the_view(self):
        tools = {t["name"]: t for t in self.request("tools/list")["result"]["tools"]}
        for name in ("slice", "make_stitch_plan", "status"):
            self.assertEqual(tools[name]["_meta"]["ui"]["resourceUri"], mcp_server.VIEW_URI)
        self.assertNotIn("_meta", tools["cancel"])

    def test_resources(self):
        listed = self.request("resources/list")["result"]["resources"]
        self.assertEqual([r["uri"] for r in listed], [mcp_server.VIEW_URI])
        read = self.request("resources/read", {"uri": mcp_server.VIEW_URI})["result"]["contents"][0]
        self.assertEqual(read["mimeType"], "text/html;profile=mcp-app")
        self.assertIn("ui/initialize", read["text"])
        self.assertNotIn("src=\"http", read["text"])               # nothing from the network
        self.assertEqual(self.request("resources/read", {"uri": "ui://x"})["error"]["code"], -32002)

    def test_slice_result_carries_the_region_map(self):
        self.set_mode("sidecar")
        reply = self.request("tools/call", {"name": "slice",
                                            "arguments": {"picture": str(self.picture)}})
        view = reply["result"]["_meta"]["digitize/view"]
        self.assertIn('<path id="f0"', view["svg"])
        self.assertEqual(view["kinds"], {"f0": "fill"})
        self.assertEqual(view["images"], ["posterized"])
        with self.measured():
            reply = self.request("tools/call", {"name": "make_stitch_plan", "arguments": {}})
        self.assertEqual(reply["result"]["_meta"]["digitize/view"]["images"], ["stitchplan"])


class SliceTool(McpTestCase):
    def test_needs_a_picture(self):
        result, message = self.call("slice")
        self.assertTrue(result["isError"])
        self.assertIn("No picture yet", message)

    def test_missing_file(self):
        result, message = self.call("slice", picture=str(self.root / "missing.png"))
        self.assertTrue(result["isError"])
        self.assertIn("no picture file", message)

    def test_slice_waits_and_reports_files(self):
        self.set_mode("sidecar")
        result, data = self.call("slice", picture=str(self.picture),
                                 overrides=[{"at_px": [12, 10], "type": "satin"}])
        self.assertFalse(result["isError"], data)
        self.assertEqual(data["state"], "idle")
        self.assertEqual(data["slice"]["slice_id"], 1)
        self.assertTrue(data["slice"]["svg"].endswith("Skyfowl-Patch/slice.svg"))
        self.assertTrue(data["slice"]["regions"].endswith("slice.regions.json"))
        self.assertEqual(data["slice"]["overrides"], [])          # what regions.json reported
        image = result["content"][1]                              # the posterized picture
        self.assertEqual(image["type"], "image")
        self.assertEqual(base64.b64decode(image["data"]), b"not a real png\n")
        job_yaml = (self.job_dir() / "job.yaml").read_text()
        self.assertIn("stitch_overrides", job_yaml)

    def test_settings_carry_to_the_next_slice(self):
        self.call("slice", picture=str(self.picture), colors=4, size_mm=80,
                  overrides=[{"at_px": [12, 10], "type": "satin"}])
        result, data = self.call("slice")
        self.assertFalse(result["isError"], data)
        self.assertEqual(data["slice"]["slice_id"], 2)
        self.assertEqual(self.tools.settings["colors"], 4)
        self.assertEqual(self.tools.settings["size_mm"], 80)
        # without regions.json the override comes back as sent, pending
        self.assertEqual(data["slice"]["overrides"],
                         [{"at_px": [12, 10], "type": "satin", "status": "pending"}])

    def test_new_picture_clears_overrides(self):
        self.call("slice", picture=str(self.picture),
                  overrides=[{"at_px": [12, 10], "type": "satin"}])
        other = self.root / "other.png"
        other.write_bytes(png_bytes())
        result, data = self.call("slice", picture=str(other))
        self.assertEqual(self.tools.settings["overrides"], [])      # E5
        self.assertEqual(data["slice"]["overrides"], [])

    def test_bad_body_is_the_servers_complaint(self):
        result, message = self.call("slice", picture=str(self.picture), colors=5)
        self.assertTrue(result["isError"])
        self.assertIn("colors must be 4, 6 or 8", message)

    def test_config_path(self):
        config = self.root / "job.yaml"
        config.write_text("fabric_color: '#222222'\n")
        result, data = self.call("slice", picture=str(self.picture), config=str(config))
        self.assertFalse(result["isError"], data)
        self.assertEqual(self.status()["slice"]["background"], "#222222")


class PlanTool(McpTestCase):
    def test_no_slice_yet(self):
        result, message = self.call("make_stitch_plan")
        self.assertTrue(result["isError"])
        self.assertIn("no Slice yet", message)

    def test_plan_with_picture(self):
        self.call("slice", picture=str(self.picture))
        with self.measured():
            result, data = self.call("make_stitch_plan")
        self.assertFalse(result["isError"], data)
        plan = data["stitchplan"]
        self.assertEqual(plan["stitches"], 17065)
        self.assertEqual(plan["removed_by_precheck"], ["s1"])
        self.assertFalse(plan["stale"])
        self.assertTrue(plan["vp3"].endswith("plan.vp3"))
        self.assertEqual(result["content"][1]["type"], "image")
        result, data = self.call("slice")
        self.assertTrue(data["stitchplan"]["stale"])                 # E7
        self.assertEqual(len(result["content"]), 1)                  # no posterized PNG here

    def test_busy_then_status_then_cancel(self):
        self.call("slice", picture=str(self.picture))
        self.set_mode("hang")
        result, data = self.call("make_stitch_plan", wait_s=0)
        self.assertEqual(data["state"], "exporting")
        self.assertIn("note", data)
        result, message = self.call("slice")
        self.assertTrue(result["isError"])
        self.assertIn("already running", message)
        result, data = self.call("status")
        self.assertEqual(data["state"], "exporting")
        result, data = self.call("cancel")
        self.assertEqual(data, {"cancelled": True})
        self.settle()
        self.assertEqual(self.status()["state"], "cancelled")


class WithPort(McpTestCase):
    def test_strict_host_check_still_passes(self):
        app = mcp_server.server.create_app(self.root, run_sh=self.run_sh, export_sh=self.export_sh,
                                           out_root=self.root / "out2", port=8799)
        tools = mcp_server.Tools(app, port=8799)
        self.assertEqual(tools.defaults["colors"], 6)


if __name__ == "__main__":
    unittest.main()
