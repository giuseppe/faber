#!/usr/bin/env python3
"""Talks to a `faber serve --a2a-keys-file` with the official A2A Python SDK
(a2a-sdk 0.3.x): the Agent Card, message/send, tasks/get, message/stream and
a follow-up in the same context.

    python3 -m venv /tmp/a2a && /tmp/a2a/bin/pip install 'a2a-sdk<0.4' httpx
    /tmp/a2a/bin/python tests/a2a_interop.py KEYS_FILE SKILL [BASE_URL]

KEYS_FILE is the --a2a-keys-file (its first key is used), SKILL a profile of
the server's database, BASE_URL http://127.0.0.1:9090 by default.
"""
import asyncio
import sys
import uuid

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart, TransportProtocol

KEYS_FILE, SKILL = sys.argv[1], sys.argv[2]
BASE = sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:9090"
KEY = open(KEYS_FILE).read().split()[1]


def message(text, **kwargs):
    return Message(
        role=Role.user,
        message_id=uuid.uuid4().hex,
        parts=[Part(root=TextPart(text=text))],
        **kwargs,
    )


def result(task):
    return [
        part.root.text if hasattr(part.root, "text") else part.root.data
        for artifact in task.artifacts or []
        for part in artifact.parts
    ]


async def main():
    headers = {"Authorization": f"Bearer {KEY}"}
    async with httpx.AsyncClient(headers=headers, timeout=120) as http:
        card = await A2ACardResolver(http, BASE).get_agent_card()
        print("card:", card.name, [s.id for s in card.skills])
        assert SKILL in [s.id for s in card.skills], "no such skill"
        for streaming in (False, True):
            config = ClientConfig(
                httpx_client=http,
                streaming=streaming,
                supported_transports=[TransportProtocol.jsonrpc],
            )
            client = ClientFactory(config).create(card)
            print("streaming" if streaming else "not streaming")
            task = None
            async for event in client.send_message(
                message("hello", metadata={"skill": SKILL})
            ):
                task, update = event
                print("  ", task.status.state.value, type(update).__name__)
            assert task.status.state.value == "completed", task
            task = await client.get_task({"id": task.id})
            print("   get_task:", task.status.state.value, result(task))
            context = task.context_id
            async for event in client.send_message(
                message("and again", context_id=context)
            ):
                task, _ = event
            assert task.context_id == context and task.status.state.value == "completed"
            print("   follow-up:", task.status.state.value, result(task))


asyncio.run(main())
