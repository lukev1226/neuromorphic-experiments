"""
Build #1 - One spiking neuron
=============================

A "leaky integrate-and-fire" (LIF) neuron, the simplest model of a real
brain cell. Think of it as a leaky bucket:

  - Input current pours water in        (integrate)
  - The bucket slowly leaks              (leak)
  - When the water hits a line -> SPIKE  (fire), then the bucket empties (reset)

The brain's energy trick: a neuron only spends energy when it spikes.
No input = no spikes = (almost) no energy. A normal AI network computes
every number on every step, whether it matters or not.

Run it:
    pip install numpy matplotlib
    python neuron.py
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Settings (try changing these!) ----------
dt = 0.1              # time step in milliseconds
total_time = 300      # how long to simulate, in milliseconds

v_rest = -70.0        # resting voltage (mV): the empty bucket
v_threshold = -55.0   # spike line (mV): fire when we reach this
v_reset = -75.0       # voltage right after a spike (slightly below rest)
tau = 10.0            # leak speed (ms): bigger = slower leak
resistance = 10.0     # how strongly input current pushes the voltage up

# ---------- 2. The input: three phases ----------
# 0-100 ms:   weak input   -> leak wins, should NOT spike
# 100-200 ms: medium input -> spikes now and then
# 200-300 ms: strong input -> spikes fast
time = np.arange(0, total_time, dt)
current = np.zeros_like(time)
current[(time >= 0) & (time < 100)] = 1.2
current[(time >= 100) & (time < 200)] = 2.0
current[(time >= 200)] = 3.5

# ---------- 3. Simulate ----------
v = v_rest
voltages = []
spike_times = []

for i, t in enumerate(time):
    # How much does the voltage change this step?
    #   leak pulls it back toward rest, input pushes it up
    dv = (-(v - v_rest) + resistance * current[i]) / tau
    v = v + dv * dt

    if v >= v_threshold:           # bucket full -> SPIKE
        spike_times.append(t)
        voltages.append(20.0)      # draw a tall spike so it's visible
        v = v_reset                # empty the bucket
    else:
        voltages.append(v)

# ---------- 4. Results ----------
print(f"Simulated {total_time} ms, {len(time)} time steps")
print(f"Total spikes: {len(spike_times)}")
for start, end, name in [(0, 100, "weak"), (100, 200, "medium"), (200, 300, "strong")]:
    n = sum(start <= s < end for s in spike_times)
    print(f"  {name:>6} input ({start}-{end} ms): {n} spikes")

print()
print("Energy idea (rough):")
print(f"  A real neuron mostly pays energy per spike: {len(spike_times)} events.")
print(f"  Your computer simulating it did {len(time)} updates, busy or not.")
print("  Brain-like (neuromorphic) chips try to only spend energy on the events.")
print("  That gap is a big part of why the brain runs on about 20 watts.")

# ---------- 5. Plot ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})

ax1.plot(time, voltages, color="tab:purple", linewidth=1)
ax1.axhline(v_threshold, color="tab:red", linestyle="--", linewidth=1, label="Spike line")
ax1.axhline(v_rest, color="gray", linestyle=":", linewidth=1, label="Resting voltage")
ax1.set_ylabel("Voltage (mV)")
ax1.set_title(f"One spiking neuron: {len(spike_times)} spikes in {total_time} ms")
ax1.legend(loc="upper left")

ax2.plot(time, current, color="tab:orange")
ax2.set_ylabel("Input")
ax2.set_xlabel("Time (ms)")
for x in (100, 200):
    ax1.axvline(x, color="lightgray", linewidth=0.8)
    ax2.axvline(x, color="lightgray", linewidth=0.8)

plt.tight_layout()
plt.savefig("neuron.png", dpi=120)
print("\nSaved plot to neuron.png")
plt.show()
