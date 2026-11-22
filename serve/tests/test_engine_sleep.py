"""The engine's own sleep: where the driver cannot suspend a process, an engine that says `sleep=engine` moves its GPU
state to host memory on SLEEP and back on WAKE, and the server only asks for RAM first and for VRAM before the wake."""
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from serve.engine import process
from serve.engine.errors import EngineDied, WakeBlocked
from serve.engine.process import ProcessEngine


class CommandEngine(ProcessEngine):
    """A ProcessEngine whose VRAM commands are recorded instead of sent."""

    def __init__(self, info=None, refuse=None):
        self.info = dict(info or {})
        self.asleep, self.parked, self.restore_s = False, None, 0.0
        self.sleep_lock = threading.Lock()
        self.proc = SimpleNamespace(pid=4321)
        self.sent, self.refuse = [], refuse

    def _vram_command(self, command, timeout):
        self.sent.append(command)
        if self.refuse is not None and command == self.refuse[0]:
            raise self.refuse[1]
        return {"vram_free_mib": 1}


def no_driver():
    return mock.patch.object(process.suspend, "available", return_value=False)


class EngineSleep(unittest.TestCase):
    def test_it_can_sleep_only_when_the_driver_or_the_engine_does_it(self):
        with no_driver():
            self.assertFalse(CommandEngine().can_sleep())
            self.assertTrue(CommandEngine({"sleep": "engine"}).can_sleep())
        with mock.patch.object(process.suspend, "available", return_value=True):
            self.assertTrue(CommandEngine().can_sleep())

    def test_sleep_asks_for_ram_first_and_keeps_what_each_card_gave_back(self):
        e = CommandEngine({"sleep": "engine"})
        free = iter([4000, 100, 29000, 100])             # GPU 0 and GPU 1, before and after
        with no_driver(), mock.patch.object(process.vram, "gpu_count", return_value=2), \
                mock.patch.object(process.vram, "free_mib", side_effect=lambda g: next(free)), \
                mock.patch.object(process.suspend, "suspend") as driver:
            e.sleep(ram_check=lambda: None)
        self.assertEqual(e.sent, ["SLEEP"])
        self.assertTrue(e.asleep)
        self.assertEqual(e.parked, {0: 25000})
        driver.assert_not_called()

    def test_a_sleep_without_the_ram_sends_nothing(self):
        e = CommandEngine({"sleep": "engine"})
        with no_driver(), self.assertRaises(ValueError):
            e.sleep(ram_check=lambda: "RAM: 10 MiB free")
        self.assertEqual((e.sent, e.asleep), ([], False))

    def test_wake_sends_wake_and_not_the_driver(self):
        e = CommandEngine({"sleep": "engine"})
        e.asleep, e.parked = True, {0: 25000}
        seen = []
        with no_driver(), mock.patch.object(process.suspend, "resume") as driver:
            e.wake(vram_check=lambda parked: seen.append(parked))
        self.assertEqual((e.sent, e.asleep, seen), (["WAKE"], False, [{0: 25000}]))
        driver.assert_not_called()

    def test_a_wake_without_the_vram_leaves_it_asleep(self):
        e = CommandEngine({"sleep": "engine"})
        e.asleep = True
        with no_driver(), self.assertRaises(WakeBlocked):
            e.wake(vram_check=lambda parked: "GPU 0 needs 25000 MiB")
        self.assertEqual((e.sent, e.asleep), ([], True))
        refused = CommandEngine({"sleep": "engine"}, refuse=("WAKE", ValueError("no device memory")))
        refused.asleep = True
        with no_driver(), self.assertRaises(WakeBlocked):
            refused.wake()
        self.assertEqual((refused.sent, refused.asleep), (["WAKE"], True))

    def test_its_state_comes_back_before_it_quits(self):
        e = CommandEngine({"sleep": "engine"})
        with no_driver():
            self.assertIsNone(e._back_for_quit())
            self.assertEqual(e.sent, ["WAKE"])
            gone = CommandEngine({"sleep": "engine"}, refuse=("WAKE", EngineDied("the engine ended")))
            self.assertEqual(gone._back_for_quit(), "the engine ended")


if __name__ == "__main__":
    unittest.main()
