"""Availability-cache staleness: behavior regression suite.

Pins the fix for the ``check_fn`` staleness class: a verdict flip
(credential appears, Docker daemon starts, OAuth login lands) never
invalidated the ``get_tool_definitions`` memo or the executor's
bridge-scope cache — the stale tool list survived until an unrelated
registry mutation. Both cache sites now key on an aggregate TTL-cached
snapshot of every probe's verdict, with a TOCTOU re-check before store.

Tests assert at public seams (``get_tool_definitions`` output, scope-cache
behavior), not private implementation details.
"""

import json
from types import SimpleNamespace


def _td(name, desc="", params=None, required=None):
    parameters = {"type": "object", "properties": params or {}}
    if required:
        parameters["required"] = required
    return {
        "type": "function",
        "function": {"name": name, "description": desc, "parameters": parameters},
    }


class TestCheckFnFlipBustsToolDefsMemo:
    """Fix 5: an availability flip propagates without a registry mutation."""

    def test_verdict_flip_changes_tool_definitions(self, monkeypatch):
        import model_tools
        from tools.registry import registry, invalidate_check_fn_cache

        available = {"value": False}

        def _flip_check():
            return available["value"]

        registry.register(
            name="flip_gated_tool",
            toolset="fliptest",
            schema=_td("flip_gated_tool", "Gated test tool.")["function"],
            handler=lambda args, **kw: json.dumps({"ok": True}),
            check_fn=_flip_check,
        )
        try:
            model_tools._clear_tool_defs_cache()
            invalidate_check_fn_cache()

            # skip_tool_search_assembly: assert on the raw exposed list —
            # deferral would otherwise (correctly) fold a plugin tool into
            # the bridge and hide the name we're asserting on. The memo
            # path under test is identical for both shapes.
            names_before = {
                t["function"]["name"]
                for t in model_tools.get_tool_definitions(
                    enabled_toolsets=["fliptest"], quiet_mode=True,
                    skip_tool_search_assembly=True,
                )
            }
            assert "flip_gated_tool" not in names_before

            # The flip: credential lands / daemon starts. No registry
            # mutation. Config untouched. Same toolsets. The TTL cache is
            # cleared the way `hermes tools enable` / config writers do.
            available["value"] = True
            invalidate_check_fn_cache()

            names_after = {
                t["function"]["name"]
                for t in model_tools.get_tool_definitions(
                    enabled_toolsets=["fliptest"], quiet_mode=True,
                    skip_tool_search_assembly=True,
                )
            }
            assert "flip_gated_tool" in names_after
        finally:
            registry.deregister("flip_gated_tool")
            model_tools._clear_tool_defs_cache()
            invalidate_check_fn_cache()

    def test_scope_cache_observes_verdict_flip(self):
        """Sibling site of the same staleness class: the executor's per-agent
        deferred-scope cache must also observe an availability flip, or the
        bridge unwrap keeps rejecting (or admitting) a tool whose probe
        verdict changed with no registry mutation."""
        from types import SimpleNamespace as NS

        import model_tools
        from agent.tool_executor import _tool_search_scoped_names
        from tools.registry import registry, invalidate_check_fn_cache

        available = {"value": False}

        registry.register(
            name="mcp__scopeflip__gated",
            toolset="mcp-scopeflip",
            schema=_td("mcp__scopeflip__gated", "Gated test tool.")["function"],
            handler=lambda args, **kw: json.dumps({"ok": True}),
            check_fn=lambda: available["value"],
        )
        agent = NS(enabled_toolsets=["mcp-scopeflip"], disabled_toolsets=None)
        try:
            model_tools._clear_tool_defs_cache()
            invalidate_check_fn_cache()

            assert "mcp__scopeflip__gated" not in _tool_search_scoped_names(agent)

            available["value"] = True
            invalidate_check_fn_cache()

            assert "mcp__scopeflip__gated" in _tool_search_scoped_names(agent)
        finally:
            registry.deregister("mcp__scopeflip__gated")
            model_tools._clear_tool_defs_cache()
            invalidate_check_fn_cache()

    def test_scope_cache_observes_config_fingerprint(self, monkeypatch, tmp_path):
        import model_tools
        from agent.tool_executor import _tool_search_scoped_names
        from tools.registry import (
            _NO_CACHE_CHECK_FNS,
            invalidate_check_fn_cache,
            no_cache_check_fn,
            registry,
        )

        config_path = tmp_path / "config.yaml"
        config_path.write_text("off", encoding="utf-8")
        monkeypatch.setattr("hermes_cli.config.get_config_path", lambda: config_path)

        def _config_check():
            return config_path.read_text(encoding="utf-8") == "enabled"

        no_cache_check_fn(_config_check)
        tool_name = "mcp__configscope__gated"
        registry.register(
            name=tool_name,
            toolset="mcp-configscope",
            schema=_td(tool_name, "Config-gated test tool.")["function"],
            handler=lambda args, **kw: json.dumps({"ok": True}),
            check_fn=_config_check,
        )
        agent = SimpleNamespace(
            enabled_toolsets=["mcp-configscope"],
            disabled_toolsets=None,
        )
        try:
            model_tools._clear_tool_defs_cache()
            invalidate_check_fn_cache()
            # Warm-up: the first rebuild can lazily register tools, which
            # flips the verdict snapshot mid-build and makes the TOCTOU
            # guard skip the store. The second call stores deterministically.
            _tool_search_scoped_names(agent)
            assert tool_name not in _tool_search_scoped_names(agent)
            cache_before = agent._tool_search_scope_cache
            assert cache_before is not None

            config_path.write_text("enabled", encoding="utf-8")
            assert tool_name in _tool_search_scoped_names(agent)
            # The config fingerprint is part of the cache key: a config write
            # must re-key the scope cache, never serve the pre-write entry.
            # Asserting the key change keeps this test red on a key that
            # omits the fingerprint even when an unrelated cache miss makes
            # the membership assert above pass by recomputation.
            assert agent._tool_search_scope_cache[0] != cache_before[0]
        finally:
            registry.deregister(tool_name)
            _NO_CACHE_CHECK_FNS.discard(_config_check)
            model_tools._clear_tool_defs_cache()
            invalidate_check_fn_cache()

    def test_snapshot_coalesces_grace_probe_and_invalidation(self, monkeypatch):
        import tools.registry as registry_module
        from tools.registry import ToolRegistry, invalidate_check_fn_cache

        local_registry = ToolRegistry()
        available = {"value": True}
        probe_calls = {"n": 0}
        clock = {"now": 1000.0}

        def _probe():
            probe_calls["n"] += 1
            return available["value"]

        monkeypatch.setattr(registry_module.time, "monotonic", lambda: clock["now"])
        local_registry.register(
            name="snapshot_gated_tool",
            toolset="snapshottest",
            schema=_td("snapshot_gated_tool", "Snapshot test tool.")["function"],
            handler=lambda args, **kw: json.dumps({"ok": True}),
            check_fn=_probe,
        )
        try:
            invalidate_check_fn_cache()
            assert local_registry.check_fn_verdict_snapshot()[0][1] is True

            available["value"] = False
            clock["now"] += registry_module._CHECK_FN_TTL_SECONDS + 1
            calls_before_grace = probe_calls["n"]
            grace_snapshots = [
                local_registry.check_fn_verdict_snapshot()
                for _ in range(8)
            ]

            assert probe_calls["n"] - calls_before_grace == 1
            assert all(snapshot == grace_snapshots[0] for snapshot in grace_snapshots)
            assert grace_snapshots[0][0][1] is True

            invalidate_check_fn_cache()
            refreshed = local_registry.check_fn_verdict_snapshot()
            assert probe_calls["n"] - calls_before_grace == 2
            assert refreshed[0][1] is False
        finally:
            invalidate_check_fn_cache()

    def test_memo_still_hits_within_ttl(self, monkeypatch):
        """The fix must not disable memoization: with stable verdicts, repeat
        calls must be served from cache (probes may run; compute must not).

        A warmup call settles lazy registration first — the first compute
        after registry churn registers dynamic tools itself, which bumps the
        generation and changes the key. That warmup miss predates this fix;
        what this test pins is that the verdict-snapshot key member does not
        introduce PERPETUAL misses.
        """
        import model_tools
        from tools.registry import registry, invalidate_check_fn_cache

        probe_calls = {"n": 0}

        def _steady_check():
            probe_calls["n"] += 1
            return True

        registry.register(
            name="steady_gated_tool",
            toolset="steadytest",
            schema=_td("steady_gated_tool", "Gated test tool.")["function"],
            handler=lambda args, **kw: json.dumps({"ok": True}),
            check_fn=_steady_check,
        )
        try:
            model_tools._clear_tool_defs_cache()
            invalidate_check_fn_cache()

            # Warmup: let lazy registration inside compute settle the
            # generation, then once more to seed the settled key.
            for _ in range(2):
                model_tools.get_tool_definitions(
                    enabled_toolsets=["steadytest"], quiet_mode=True)

            compute_calls = {"n": 0}
            real_compute = model_tools._compute_tool_definitions

            def _counting_compute(*args, **kwargs):
                compute_calls["n"] += 1
                return real_compute(*args, **kwargs)

            monkeypatch.setattr(model_tools, "_compute_tool_definitions", _counting_compute)

            first = model_tools.get_tool_definitions(
                enabled_toolsets=["steadytest"], quiet_mode=True)
            second = model_tools.get_tool_definitions(
                enabled_toolsets=["steadytest"], quiet_mode=True)

            assert compute_calls["n"] == 0
            assert [t["function"]["name"] for t in first] == \
                   [t["function"]["name"] for t in second]
        finally:
            registry.deregister("steady_gated_tool")
            model_tools._clear_tool_defs_cache()
            invalidate_check_fn_cache()
