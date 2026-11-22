"""Real Localhost HTTP/1.1 Chunked-SSE Disaggregated AFD Serving Harness.

Validates SonarMoE and OblivDispatch over actual macOS kernel TCP/IP sockets (`127.0.0.1`,
`TCP_NODELAY`) with zero shared KV cache between Tenant A and Tenant B:
  - Tenant A endpoint (`/v1/tenant_a/dispatch`): Sets isolated Victim A expert batch state.
  - Tenant B endpoint (`/v1/tenant_b/chat/completions`, `text/event-stream` SSE):
    Executes REAL Apple M5 Pro MPS FP16 expert MLP kernels (`gate_up_proj -> SiLU -> down_proj`
    synchronized on Metal GPU!) on the co-batched bottleneck Expert Rank before flushing
    each SSE token chunk over the TCP socket.
  - External Attacker Client B connects over a separate `http.client.HTTPConnection` socket
    and records ONLY client-side `time.perf_counter_ns()` SSE chunk inter-arrival times!
"""

from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sonarmoe.models import MoEArchSpec, get_experts_module


class _SharedExpertPoolState:
    def __init__(
        self,
        model: Any,
        spec: MoEArchSpec,
        canary_counts: np.ndarray,
        max_tokens: int = 224,
    ):
        self.model = model
        self.spec = spec
        self.canary_counts = canary_counts  # (M, L, E)
        self.max_tokens = max_tokens
        self.lock = threading.Lock()
        self.gpu_lock = threading.Lock()
        self.active_victim_counts = np.zeros(
            (spec.num_layers, spec.num_experts), dtype=np.int32
        )
        device = next(model.parameters()).device
        hidden_dim = model.config.hidden_size
        self.x_buf = torch.randn(max_tokens, hidden_dim, device=device, dtype=torch.float16)

    @torch.no_grad()
    def execute_real_mps_step_for_probe(self, probe_idx: int) -> None:
        """Executes real MPS FP16 expert GEMMs for the probed rank across layers."""
        with self.lock:
            vic = self.active_victim_counts.copy()

        can = self.canary_counts[probe_idx]
        tot = np.minimum(vic + can, self.max_tokens)
        e_per_r = self.spec.experts_per_rank

        # Identify the bottleneck rank p* at each layer and execute its exact active expert MLPs
        # on the real Apple M5 Pro Metal GPU before emitting the streamed HTTP token!
        rank_token_sums = tot.reshape(
            self.spec.num_layers, self.spec.num_ranks, e_per_r
        ).sum(axis=-1)
        bottleneck_ranks = rank_token_sums.argmax(axis=-1)  # (L,)

        with self.gpu_lock:
            for l_idx in range(self.spec.num_layers):
                r_star = int(bottleneck_ranks[l_idx])
                e_start = r_star * e_per_r
                exp_mod = get_experts_module(self.model.model.layers[l_idx])
                for e_idx in range(e_start, e_start + e_per_r):
                    n_tok = int(tot[l_idx, e_idx])
                    if n_tok > 0:
                        x_s = self.x_buf[:n_tok]
                        gu = F.linear(x_s, exp_mod.gate_up_proj[e_idx])
                        g, u = gu.chunk(2, dim=-1)
                        _ = F.linear(F.silu(g) * u, exp_mod.down_proj[e_idx])
            torch.mps.synchronize()


class _TCPNoDelayHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def get_request(self):
        conn, addr = super().get_request()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return conn, addr


def _make_handler(pool: _SharedExpertPoolState):
    class _AFDHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return  # Silence per-request stdout logging

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            payload = json.loads(raw_body.decode("utf-8"))

            if self.path == "/v1/tenant_a/dispatch":
                counts = np.asarray(payload["expert_counts"], dtype=np.int32)
                with pool.lock:
                    pool.active_victim_counts = counts
                resp = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                self.wfile.flush()
                return

            if self.path == "/v1/tenant_b/chat/completions":
                probe_idx = int(payload["probe_idx"])
                num_tokens = int(payload.get("max_tokens", 3))

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send_sse_chunk(obj: dict[str, Any]) -> None:
                    data_bytes = f"data: {json.dumps(obj)}\n\n".encode("utf-8")
                    chunk_hdr = f"{len(data_bytes):X}\r\n".encode("ascii")
                    self.wfile.write(chunk_hdr + data_bytes + b"\r\n")
                    self.wfile.flush()

                # Initial TTFT anchor frame
                send_sse_chunk({"tok": 0, "anchor": True})
                for step_k in range(1, num_tokens + 1):
                    pool.execute_real_mps_step_for_probe(probe_idx)
                    send_sse_chunk({"tok": step_k, "probe": probe_idx})

                # Terminating chunk
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                return

            self.send_response(404)
            self.end_headers()

    return _AFDHandler


def _read_chunked_sse_itl_us(conn: http.client.HTTPConnection, path: str, body: bytes) -> float:
    """Client-side HTTP/1.1 chunked SSE reader recording exact arrival timestamps."""
    conn.request(
        "POST",
        path,
        body=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "keep-alive",
        },
    )
    resp = conn.getresponse()
    assert resp.status == 200

    # Read SSE `data: {...}\n\n` frames through HTTPResponse's native chunked decoder
    timestamps_ns: list[int] = []
    while True:
        line = resp.readline()
        if not line:
            break
        if line.startswith(b"data:"):
            _blank = resp.readline()
            timestamps_ns.append(time.perf_counter_ns())
    resp.close()

    # Compute lower-median Inter-Token Latency across streamed tokens after anchor frame
    if len(timestamps_ns) < 2:
        return 0.0
    itls_us = [
        (timestamps_ns[i] - timestamps_ns[i - 1]) / 1000.0
        for i in range(1, len(timestamps_ns))
    ]
    return float(sorted(itls_us)[len(itls_us) // 2])


def run_live_http_loopback_benchmark(
    model: Any,
    spec: MoEArchSpec,
    victim_counts: np.ndarray,   # (N_q, L, E)
    canary_counts: np.ndarray,   # (M, L, E)
    stream_tokens_per_probe: int = 3,
) -> dict[str, Any]:
    """Runs end-to-end localhost TCP/HTTP SSE streaming attack measurements on MPS."""
    t0 = time.perf_counter()
    pool = _SharedExpertPoolState(model, spec, canary_counts)
    server = _TCPNoDelayHTTPServer(("127.0.0.1", 0), _make_handler(pool))
    port = server.server_address[1]

    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    num_q = victim_counts.shape[0]
    num_m = canary_counts.shape[0]
    observed_delta_itl = np.zeros((num_q, num_m), dtype=np.float32)

    try:
        conn_victim = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn_attacker = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn_victim.connect()
        conn_attacker.connect()
        conn_victim.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn_attacker.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # Warmup HTTP + MPS pipeline and calibrate empty-victim baseline ITL_0(m)
        zero_vic = np.zeros((spec.num_layers, spec.num_experts), dtype=np.int32)
        conn_victim.request(
            "POST",
            "/v1/tenant_a/dispatch",
            body=json.dumps({"expert_counts": zero_vic.tolist()}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Connection": "keep-alive"},
        )
        _ = conn_victim.getresponse().read()

        for warm_m in range(min(4, num_m)):
            _ = _read_chunked_sse_itl_us(
                conn_attacker,
                "/v1/tenant_b/chat/completions",
                json.dumps({"probe_idx": warm_m, "max_tokens": 2}).encode("utf-8"),
            )

        baseline_itl_us = np.zeros(num_m, dtype=np.float32)
        for m_idx in range(num_m):
            req_b = json.dumps(
                {"probe_idx": m_idx, "max_tokens": stream_tokens_per_probe}
            ).encode("utf-8")
            baseline_itl_us[m_idx] = _read_chunked_sse_itl_us(
                conn_attacker, "/v1/tenant_b/chat/completions", req_b
            )

        # Measure real client-side HTTP SSE ITL across all victim queries x canary probes
        probe_payloads = [
            json.dumps(
                {"probe_idx": m_idx, "max_tokens": stream_tokens_per_probe}
            ).encode("utf-8")
            for m_idx in range(num_m)
        ]

        for q_idx in range(num_q):
            vic_body = json.dumps(
                {"expert_counts": victim_counts[q_idx].tolist()}
            ).encode("utf-8")
            conn_victim.request(
                "POST",
                "/v1/tenant_a/dispatch",
                body=vic_body,
                headers={"Content-Type": "application/json", "Connection": "keep-alive"},
            )
            _ = conn_victim.getresponse().read()

            for m_idx in range(num_m):
                itl_us = _read_chunked_sse_itl_us(
                    conn_attacker,
                    "/v1/tenant_b/chat/completions",
                    probe_payloads[m_idx],
                )
                observed_delta_itl[q_idx, m_idx] = itl_us - baseline_itl_us[m_idx]

        conn_victim.close()
        conn_attacker.close()
    finally:
        server.shutdown()
        server.server_close()

    elapsed = round(time.perf_counter() - t0, 2)
    return {
        "delta_itl_us": observed_delta_itl,
        "baseline_probe_itl_mean_ms": round(float(baseline_itl_us.mean()) / 1000.0, 3),
        "baseline_probe_itl_std_us": round(float(baseline_itl_us.std()), 2),
        "total_http_sse_frames_streamed": int(
            (num_q + 1) * num_m * (stream_tokens_per_probe + 1)
        ),
        "wall_time_sec": elapsed,
    }
