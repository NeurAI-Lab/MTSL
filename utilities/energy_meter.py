import sys
from time import time, sleep
from threading import Timer
import pynvml

from tensorboardX import SummaryWriter


class EnergyMeter:
    def __init__(self, writer: SummaryWriter = None, period=0.01):
        assert period >= 0.005, "Measurement period below than 5ms"
        self.period = period
        pynvml.nvmlInit()
        self.gpu_handles = [
            pynvml.nvmlDeviceGetHandleByIndex(idx)
            for idx in range(pynvml.nvmlDeviceGetCount())
        ]
        self.writer = writer
        if self.writer is not None:
            self.writer.add_scalar("xtras/energy_usage", 0, 0)
        self.done = False
        self.steps = 0
        self.energy = 0
        self.next_t = 0
        self.t = None

    def __enter__(self):
        self.done = False
        self.steps = 0
        self.energy = 0
        self.next_t = time()
        self.run()
        return self

    def _get_energy_usage(self):
        energy = 0
        for handle in self.gpu_handles:
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            energy += power / 1000.0 * self.period
        return energy

    def run(self):
        if not self.done:
            self.t = Timer(self.next_t - time(), self.run)
            self.t.start()
            self.next_t += self.period
            self.steps = self.steps + 1
            self.energy = self.energy + self._get_energy_usage()
            if self.steps % 100 == 0:
                if self.writer is not None:
                    self.writer.add_scalar(
                        "xtras/energy_usage", self.energy, self.steps
                    )

    def __exit__(self, type, value, traceback):
        self.done = True
        self.t.cancel()
        print(f"Total energy used: {int(self.energy)} J")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        PERIOD = float(sys.argv[1])
    else:
        PERIOD = 0.01

    em = EnergyMeter(PERIOD)
    with em:
        # put code you what to measure energy of here
        sleep(2)
