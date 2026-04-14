"""
ARGUS Distributed Consistency Module
Task 2: Lamport Clocks + Vector Clocks + Causal Consistency Verification

Simulates 3 processes exchanging messages, verifies causal ordering,
and saves results to results/latency/distributed_sync.csv.
"""

import os
import pandas as pd
import numpy as np

os.makedirs("results/latency", exist_ok=True)


# ============================================================
# Lamport Clock
# ============================================================

class LamportClock:
    """
    Lamport Logical Clock for causal consistency.

    Rules:
      tick()    : local event  -> time += 1
      send(msg) : send event   -> time += 1, attach timestamp
      receive() : recv event   -> time = max(local, msg.ts) + 1

    Guarantee: if A -> B (A happened-before B), then LC(A) < LC(B).
    """
    def __init__(self, process_id: int):
        self.pid  = process_id
        self.time = 0

    def tick(self) -> int:
        self.time += 1
        return self.time

    def send(self, msg: dict) -> dict:
        self.time += 1
        msg["lamport_ts"] = self.time
        return msg

    def receive(self, msg: dict) -> int:
        self.time = max(self.time, msg["lamport_ts"]) + 1
        return self.time


# ============================================================
# Vector Clock
# ============================================================

class VectorClock:
    """
    Vector Logical Clock for causal consistency.

    Rules:
      send(pid, msg) : clock[pid] += 1, attach full vector
      receive(msg)   : element-wise max, then clock[pid] += 1

    Guarantee: vc1 -> vc2  iff  vc1[i] <= vc2[i] for all i
               AND vc1[i]  <  vc2[i] for at least one i.
    """
    def __init__(self, process_id: int, n_processes: int):
        self.pid   = process_id
        self.n     = n_processes
        self.clock = [0] * n_processes

    def tick(self) -> list:
        self.clock[self.pid] += 1
        return self.clock[:]

    def send(self, msg: dict) -> dict:
        self.clock[self.pid] += 1
        msg["vector_clock"] = self.clock[:]
        return msg

    def receive(self, msg: dict) -> list:
        received = msg["vector_clock"]
        for i in range(self.n):
            self.clock[i] = max(self.clock[i], received[i])
        self.clock[self.pid] += 1
        return self.clock[:]

    @staticmethod
    def happens_before(vc1: list, vc2: list) -> bool:
        """True if vc1 causally precedes vc2."""
        return (all(vc1[i] <= vc2[i] for i in range(len(vc1))) and
                any(vc1[i] <  vc2[i] for i in range(len(vc1))))

    @staticmethod
    def concurrent(vc1: list, vc2: list) -> bool:
        """True if vc1 and vc2 are causally independent."""
        return (not VectorClock.happens_before(vc1, vc2) and
                not VectorClock.happens_before(vc2, vc1))


# ============================================================
# Distributed Simulation
# ============================================================

def run_distributed_simulation() -> pd.DataFrame:
    """
    Simulate 3 processes exchanging 10 messages each with Lamport and
    Vector clocks.  Verify causal ordering after every receive event.

    Causal order check (Lamport):
        recv_ts > send_ts   (strict: receive happens after send)

    Saves: results/latency/distributed_sync.csv
    Columns: event, process, lamport_ts, vector_clock, causal_order_ok
    """
    print("=" * 60)
    print("  ARGUS: Distributed Causal Consistency Simulation")
    print("=" * 60)

    N_PROCS    = 3
    N_MSGS     = 10
    np.random.seed(42)

    lamport = [LamportClock(i)          for i in range(N_PROCS)]
    vector  = [VectorClock(i, N_PROCS)  for i in range(N_PROCS)]

    records = []

    for msg_num in range(N_MSGS):
        for sender in range(N_PROCS):
            # Pick a different receiver
            receiver = (sender + 1 + np.random.randint(0, N_PROCS - 1)) % N_PROCS

            msg = {
                "sender":  sender,
                "receiver": receiver,
                "payload": f"m{msg_num}_{sender}",
            }

            # --- SEND ---
            lamport[sender].send(msg)
            vector[sender].send(msg)

            send_lts = msg["lamport_ts"]
            send_vc  = msg["vector_clock"][:]

            records.append({
                "event":          f"SEND_{msg_num}_P{sender}->P{receiver}",
                "process":        sender,
                "lamport_ts":     send_lts,
                "vector_clock":   str(send_vc),
                "causal_order_ok": True,   # send events are trivially valid
            })

            # --- RECEIVE ---
            recv_lts = lamport[receiver].receive(msg)
            recv_vc  = vector[receiver].receive(msg)

            # Lamport causality: recv timestamp must be strictly greater than send
            causal_ok = recv_lts > send_lts

            records.append({
                "event":          f"RECV_{msg_num}_P{sender}->P{receiver}",
                "process":        receiver,
                "lamport_ts":     recv_lts,
                "vector_clock":   str(recv_vc),
                "causal_order_ok": causal_ok,
            })

            if not causal_ok:
                print(f"  [VIOLATION] msg {msg_num}: send_ts={send_lts}  recv_ts={recv_lts}")

    df = pd.DataFrame(records)
    df.to_csv("results/latency/distributed_sync.csv", index=False)

    n_violations = int((~df["causal_order_ok"]).sum())
    print(f"\nTotal events:        {len(df)}")
    print(f"Causality violations: {n_violations}")
    print(f"Saved: results/latency/distributed_sync.csv")

    # Final clock states
    print("\nFinal Lamport clocks:")
    for i in range(N_PROCS):
        print(f"  P{i}: {lamport[i].time}")
    print("Final Vector clocks:")
    for i in range(N_PROCS):
        print(f"  P{i}: {vector[i].clock}")

    return df


if __name__ == "__main__":
    run_distributed_simulation()
