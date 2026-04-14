/*
 * ARGUS Microkernel: Lock-Free IPC
 * Extended with MPMC queue (Dmitry Vyukov's algorithm),
 * causal Lamport timestamps, and throughput benchmark.
 *
 * Compile:
 *   gcc -O3 -std=c11 -lpthread src/c/argus_ipc.c -o src/c/argus_ipc
 *
 * Run:
 *   ./src/c/argus_ipc
 */

#define _XOPEN_SOURCE 600

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

/* =========================================================
 * MODULE 1: Original SPSC Ring Buffer (preserved)
 * ========================================================= */

#define QUEUE_SIZE 1024

typedef struct {
    int        buffer[QUEUE_SIZE];
    _Atomic int head;
    _Atomic int tail;
} LockFreeQueue;

static LockFreeQueue ipc_queue = { .head = 0, .tail = 0 };

/* Global Lamport clock for causal ordering */
static _Atomic long global_lamport = 0;

static long lamport_tick(void) {
    return atomic_fetch_add_explicit(&global_lamport, 1, memory_order_seq_cst) + 1;
}

static long lamport_recv(long sender_ts) {
    long cur;
    do {
        cur = atomic_load_explicit(&global_lamport, memory_order_relaxed);
        if (sender_ts < cur) break;
    } while (!atomic_compare_exchange_weak_explicit(
                 &global_lamport, &cur, sender_ts + 1,
                 memory_order_seq_cst, memory_order_relaxed));
    return atomic_load_explicit(&global_lamport, memory_order_seq_cst);
}

/* Causal message send (SPSC, lock-free) */
static void send_message(int process_id, int message) {
    int current_tail = atomic_load(&ipc_queue.tail);
    int next_tail    = (current_tail + 1) % QUEUE_SIZE;

    if (next_tail != atomic_load(&ipc_queue.head)) {
        long ts = lamport_tick();
        ipc_queue.buffer[current_tail] = message;
        atomic_store(&ipc_queue.tail, next_tail);
        printf("[Process %d] Sent Causal Message: %d  (Lamport ts=%ld)\n",
               process_id, message, ts);
    } else {
        printf("[Process %d] IPC Queue Full! Message Dropped.\n", process_id);
    }
}

static void *process_thread(void *arg) {
    int pid = *((int *)arg);
    for (int i = 0; i < 3; i++) {
        send_message(pid, (pid * 10) + i);
        usleep(1000);
    }
    return NULL;
}

/* =========================================================
 * MODULE 2: MPMC Queue — Dmitry Vyukov's Sequence-Based Algorithm
 *
 * Each cell carries:
 *   - sequence: used to coordinate producer/consumer slots (CAS-free slot grab)
 *   - sender_id:   which producer sent this message
 *   - lamport_ts:  causal timestamp at send time
 *   - payload:     message content
 *
 * Correctness: ABA-free via monotonically increasing sequence numbers.
 * ========================================================= */

#define MPMC_SIZE  4096   /* must be power of 2 */
#define CACHE_LINE 64

typedef struct {
    _Atomic long sequence;
    int          sender_id;
    long         lamport_ts;
    int          payload;
    char         _pad[CACHE_LINE - sizeof(_Atomic long) - sizeof(int)
                       - sizeof(long) - sizeof(int)];
} MPMCCell;

typedef struct {
    MPMCCell     buffer[MPMC_SIZE];
    char         _pad0[CACHE_LINE];
    _Atomic long head;
    char         _pad1[CACHE_LINE - sizeof(_Atomic long)];
    _Atomic long tail;
    char         _pad2[CACHE_LINE - sizeof(_Atomic long)];
} MPMCQueue;

static void mpmc_init(MPMCQueue *q) {
    memset(q, 0, sizeof(*q));
    for (long i = 0; i < MPMC_SIZE; i++)
        atomic_store_explicit(&q->buffer[i].sequence, i, memory_order_relaxed);
    atomic_store_explicit(&q->head, 0, memory_order_relaxed);
    atomic_store_explicit(&q->tail, 0, memory_order_relaxed);
}

static bool mpmc_enqueue(MPMCQueue *q, int sender, long ts, int payload) {
    MPMCCell *cell;
    long pos = atomic_load_explicit(&q->tail, memory_order_relaxed);

    for (;;) {
        cell      = &q->buffer[pos & (MPMC_SIZE - 1)];
        long seq  = atomic_load_explicit(&cell->sequence, memory_order_acquire);
        long diff = seq - pos;

        if (diff == 0) {
            /* Slot is free — try to claim it */
            if (atomic_compare_exchange_weak_explicit(
                    &q->tail, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                break;   /* we own this slot */
            }
        } else if (diff < 0) {
            return false;   /* queue is full */
        } else {
            pos = atomic_load_explicit(&q->tail, memory_order_relaxed);
        }
    }

    cell->sender_id  = sender;
    cell->lamport_ts = ts;
    cell->payload    = payload;
    atomic_store_explicit(&cell->sequence, pos + 1, memory_order_release);
    return true;
}

static bool mpmc_dequeue(MPMCQueue *q, int *sender, long *ts, int *payload) {
    MPMCCell *cell;
    long pos = atomic_load_explicit(&q->head, memory_order_relaxed);

    for (;;) {
        cell      = &q->buffer[pos & (MPMC_SIZE - 1)];
        long seq  = atomic_load_explicit(&cell->sequence, memory_order_acquire);
        long diff = seq - (pos + 1);

        if (diff == 0) {
            /* Slot is ready to consume — try to claim it */
            if (atomic_compare_exchange_weak_explicit(
                    &q->head, &pos, pos + 1,
                    memory_order_relaxed, memory_order_relaxed)) {
                break;   /* we own this slot */
            }
        } else if (diff < 0) {
            return false;   /* queue is empty */
        } else {
            pos = atomic_load_explicit(&q->head, memory_order_relaxed);
        }
    }

    *sender  = cell->sender_id;
    *ts      = cell->lamport_ts;
    *payload = cell->payload;
    atomic_store_explicit(&cell->sequence, pos + MPMC_SIZE, memory_order_release);

    /* Causal receive: update local Lamport clock */
    lamport_recv(*ts);
    return true;
}

/* =========================================================
 * MODULE 3: MPMC Benchmark
 * ========================================================= */

#define BENCH_MSGS  50000    /* messages per producer */

typedef struct {
    MPMCQueue   *q;
    int          producer_id;
    long         n_messages;
    _Atomic long *sent_count;
} ProducerArg;

typedef struct {
    MPMCQueue   *q;
    long         n_expected;
    _Atomic long *recv_count;
    _Atomic int  *stop;
} ConsumerArg;

static void *producer_fn(void *arg) {
    ProducerArg *a = (ProducerArg *)arg;
    for (long i = 0; i < a->n_messages; i++) {
        long ts = lamport_tick();
        while (!mpmc_enqueue(a->q, a->producer_id, ts, (int)(i & 0x7fffffff)))
            ; /* spin until space available */
        atomic_fetch_add_explicit(a->sent_count, 1, memory_order_relaxed);
    }
    return NULL;
}

static void *consumer_fn(void *arg) {
    ConsumerArg *a = (ConsumerArg *)arg;
    int  sender; long ts; int payload;
    while (atomic_load_explicit(a->recv_count, memory_order_relaxed) < a->n_expected) {
        if (mpmc_dequeue(a->q, &sender, &ts, &payload))
            atomic_fetch_add_explicit(a->recv_count, 1, memory_order_relaxed);
    }
    return NULL;
}

static double bench_mpmc(int n_producers, int n_consumers) {
    MPMCQueue *q = (MPMCQueue *)malloc(sizeof(MPMCQueue));
    mpmc_init(q);

    long total_msgs = (long)n_producers * BENCH_MSGS;

    _Atomic long sent_count = 0;
    _Atomic long recv_count = 0;
    _Atomic int  stop_flag  = 0;

    ProducerArg *pargs = malloc(sizeof(ProducerArg) * (size_t)n_producers);
    ConsumerArg *cargs = malloc(sizeof(ConsumerArg) * (size_t)n_consumers);
    pthread_t   *ptids = malloc(sizeof(pthread_t)   * (size_t)n_producers);
    pthread_t   *ctids = malloc(sizeof(pthread_t)   * (size_t)n_consumers);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < n_producers; i++) {
        pargs[i] = (ProducerArg){q, i, BENCH_MSGS, &sent_count};
        pthread_create(&ptids[i], NULL, producer_fn, &pargs[i]);
    }
    for (int i = 0; i < n_consumers; i++) {
        /* n_expected = total_msgs so all consumers together drain the queue */
        cargs[i] = (ConsumerArg){q, total_msgs, &recv_count, &stop_flag};
        pthread_create(&ctids[i], NULL, consumer_fn, &cargs[i]);
    }

    for (int i = 0; i < n_producers; i++) pthread_join(ptids[i], NULL);
    for (int i = 0; i < n_consumers; i++) pthread_join(ctids[i], NULL);

    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed_s = (t1.tv_sec - t0.tv_sec) +
                       (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    double throughput = (double)total_msgs / elapsed_s;

    free(q); free(pargs); free(cargs); free(ptids); free(ctids);
    return throughput;
}

/* =========================================================
 * main()
 * ========================================================= */

int main(void) {
    printf("===========================================\n");
    printf("  ARGUS MICROKERNEL: LOCK-FREE IPC BOOT    \n");
    printf("===========================================\n\n");

    /* --- Legacy SPSC demo --- */
    pthread_t p1, p2;
    int pid1 = 1, pid2 = 2;
    pthread_create(&p1, NULL, process_thread, &pid1);
    pthread_create(&p2, NULL, process_thread, &pid2);
    pthread_join(p1, NULL);
    pthread_join(p2, NULL);
    printf("\n[KERNEL] SPSC IPC: %d messages routed without locks.\n\n",
           atomic_load(&ipc_queue.tail));

    /* --- MPMC Benchmark --- */
    printf("[KERNEL] Running MPMC Throughput Benchmark...\n");
    printf("  (each config sends %d messages per producer)\n\n", BENCH_MSGS);

    int configs[][2] = {{1,1}, {2,2}, {4,4}};
    int n_configs = 3;

    double results[3];
    for (int i = 0; i < n_configs; i++) {
        int np = configs[i][0], nc = configs[i][1];
        /* reset Lamport clock between runs */
        atomic_store(&global_lamport, 0);
        results[i] = bench_mpmc(np, nc);
        printf("  %dP/%dC : %.0f msg/sec  (%.2f M/s)\n",
               np, nc, results[i], results[i] / 1e6);
    }

    /* --- Write CSV --- */
    /* Ensure directory exists via mkdir -p equivalent */
    system("mkdir -p results/latency");

    FILE *fp = fopen("results/latency/ipc_mpmc_benchmark.csv", "w");
    if (!fp) {
        fprintf(stderr, "[ERROR] Cannot open results/latency/ipc_mpmc_benchmark.csv\n");
        return 1;
    }
    fprintf(fp, "config,n_producers,n_consumers,messages_per_producer,"
                "total_messages,throughput_msg_per_sec\n");
    for (int i = 0; i < n_configs; i++) {
        int np = configs[i][0], nc = configs[i][1];
        long total = (long)np * BENCH_MSGS;
        fprintf(fp, "%dP%dC,%d,%d,%d,%ld,%.0f\n",
                np, nc, np, nc, BENCH_MSGS, total, results[i]);
    }
    fclose(fp);
    printf("\nSaved: results/latency/ipc_mpmc_benchmark.csv\n");

    return 0;
}
