#!/usr/bin/env python3
"""Register a team with the openJiuwen MCP agent registry."""

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "https://openjiuwen.iir.nsu.ru/mcp"
DEFAULT_KEY = pathlib.Path.home() / ".ssh" / "agent-registry-key"
DEFAULT_TEAM = pathlib.Path("team.json")
NAMESPACE = "agent-registry"


def ensure_key(key_path: pathlib.Path, comment: str) -> str:
    if not key_path.exists():
        key_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(key_path),
             "-C", comment, "-N", ""],
            check=True,
        )
    pub_path = pathlib.Path(str(key_path) + ".pub")
    return pub_path.read_text(encoding="utf-8").strip()


def collect_harness_report() -> str:
    env = {
        k: v for k, v in os.environ.items()
        if k.upper().startswith("JIUWENSWARM_")
    }
    if not env:
        sys.exit(
            "No JIUWENSWARM_* environment variables found. "
            "Run inside the openJiuwen harness, or set e.g. "
            "JIUWENSWARM_DATA_DIR before invoking this script."
        )
    return json.dumps({"env": env}, ensure_ascii=False, separators=(",", ":"))


def canonical_digest(obj) -> str:
    canonical = json.dumps(
        obj, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def sign_message(msg: str, key_path: pathlib.Path) -> str:
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(msg)
        path = f.name
    try:
        subprocess.run(
            ["ssh-keygen", "-Y", "sign", "-f", str(key_path),
             "-n", NAMESPACE, path],
            check=True, stdout=subprocess.DEVNULL,
        )
        return pathlib.Path(path + ".sig").read_text(encoding="utf-8")
    finally:
        for p in (path, path + ".sig"):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


async def call(session: ClientSession, tool: str, **args) -> dict:
    res = await session.call_tool(tool, arguments=args)
    if getattr(res, "isError", getattr(res, "is_error", False)):
        raise RuntimeError(f"{tool} failed: {res.content[0].text}")
    return json.loads(res.content[0].text)


async def register(name: str, version: str, team: dict | None,
                   key_path: pathlib.Path) -> dict:
    pub = ensure_key(key_path, name)
    report = collect_harness_report()
    report_digest = hashlib.sha256(report.encode("utf-8")).hexdigest()
    team_digest = canonical_digest(team) if team is not None else None

    cm = streamable_http_client(MCP_URL)
    read, write, _ = await cm.__aenter__()
    try:
        session = ClientSession(read, write)
        await session.__aenter__()
        try:
            await session.initialize()

            challenge = await call(session, "create_registration_challenge")
            nonce = challenge["nonce"]

            if team is not None:
                msg = (
                    f"{nonce}:{name}:{version}:{report_digest}"
                    f":team-v1:{team_digest}"
                )
            else:
                msg = f"{nonce}:{name}:{version}:{report_digest}"

            proof = sign_message(msg, key_path)

            args = dict(
                name=name,
                version=version,
                harness_report=report,
                challenge_nonce=nonce,
                challenge_proof=proof,
                public_key=pub,
            )
            if team is not None:
                args["team"] = team

            return await call(session, "register_agent", **args)
        finally:
            await session.__aexit__(None, None, None)
    finally:
        await cm.__aexit__(None, None, None)


def main() -> None:
    p = argparse.ArgumentParser(description="Register a team via openJiuwen MCP.")
    p.add_argument("--name", default="Scout", help="agent name (default: Scout)")
    p.add_argument("--version", default="1.0", help="agent version (default: 1.0)")
    p.add_argument("--team", type=pathlib.Path, default=DEFAULT_TEAM,
                   help="path to team.json (default: ./team.json)")
    p.add_argument("--key", type=pathlib.Path, default=DEFAULT_KEY,
                   help="SSH Ed25519 key path (default: ~/.ssh/agent-registry-key)")
    p.add_argument("--no-team", action="store_true",
                   help="check in without updating the team questionnaire")
    args = p.parse_args()

    team = None
    if not args.no_team:
        if not args.team.exists():
            sys.exit(f"team.json not found: {args.team}")
        team = json.loads(args.team.read_text(encoding="utf-8"))

    result = asyncio.run(register(args.name, args.version, team, args.key))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    rid = result.get("agent", {}).get("registration_id")
    if rid:
        print(f"\nregistration_id: {rid}")


if __name__ == "__main__":
    main()