#![no_std]
#![no_main]

use core::panic::PanicInfo;
use core::sync::atomic::{AtomicUsize, Ordering};

// ==========================================
// MODULE 3: DISTRIBUTED MEMORY (DS)
// Υλοποίηση Lamport Clock για Causal Consistency
// ==========================================
static LAMPORT_CLOCK: AtomicUsize = AtomicUsize::new(0);

pub fn increment_clock() -> usize {
    // Lock-free αύξηση του ρολογιού
    LAMPORT_CLOCK.fetch_add(1, Ordering::SeqCst)
}

pub fn sync_clock(received_time: usize) {
    let mut current = LAMPORT_CLOCK.load(Ordering::SeqCst);
    while received_time > current {
        // Lock-free συγχρονισμός με το ρολόι του δικτύου/άλλου node
        match LAMPORT_CLOCK.compare_exchange_weak(
            current,
            received_time + 1,
            Ordering::SeqCst,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(x) => current = x,
        }
    }
}

// ==========================================
// MODULE 1: MICROKERNEL CORE (OS)
// ==========================================
// Entry point του πυρήνα (καλείται από τον Assembly Bootloader)
#[no_mangle]
pub extern "C" fn _start() -> ! {
    // 1. Εδώ θα γινόταν το GDT/IDT setup
    
    // 2. Ενεργοποίηση του Lamport Clock για το Node
    let boot_time = increment_clock();
    
    // 3. Εδώ το λειτουργικό παραδίδει τον έλεγχο στον C IPC handler 
    // και τον Python ARGUS-SYNC Scheduler
    
    loop {
        // Ασφαλής ατέρμονος βρόχος του Microkernel (Panoptes idle loop)
    }
}

// Υποχρεωτικό για no_std περιβάλλοντα (Τι γίνεται αν κρασάρει ο πυρήνας;)
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}