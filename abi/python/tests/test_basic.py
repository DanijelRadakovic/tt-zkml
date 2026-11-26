# SPDX-FileCopyrightText: © 2026 Danijel Radakovic
#
# SPDX-License-Identifier: MIT
import math
import random
import unittest

import tt_zkml


def reference_layer(x, w, m, k, n):
    """One quantized layer on the host (no activation): the definition the device implements."""

    def scale(v):
        return math.ldexp(1.0, tt_zkml.pot_k(max(abs(e) for e in v), 8))

    sx, sw = scale(x), scale(w)
    qx = [round(e * sx) for e in x]  # Python's round is round half to even
    qw = [round(e * sw) for e in w]
    acc = [sum(qx[i * k + kk] * qw[kk * n + j] for kk in range(k)) for i in range(m) for j in range(n)]
    amax = max(abs(a) for a in acc)
    ko = 127
    if amax:
        t = amax.bit_length()
        ko = 7 - t
        if ko < 0 and amax > (127 << -ko):
            ko -= 1
    out = []
    for a in acc:
        mag = abs(a)
        if ko >= 0:
            q = 0 if ko > 40 else mag << ko
        else:
            s = -ko
            q = (mag + (1 << (s - 1)) - 1 + ((mag >> s) & 1)) >> s
        out.append((0x80 | q) if (a < 0 and q) else q)
    return sx, sw, math.ldexp(1.0, ko), acc, bytes(out)


class Host(unittest.TestCase):
    def test_versions(self):
        self.assertEqual(tt_zkml.abi_version(), tt_zkml.ABI_VERSION)
        self.assertRegex(tt_zkml.version(), r"^\d+\.\d+\.\d+$")

    def test_pot_k(self):
        self.assertEqual(tt_zkml.pot_k(1.0, 8), 6)
        self.assertEqual(tt_zkml.pot_k(127.0, 8), 0)
        self.assertEqual(tt_zkml.pot_k(128.0, 8), -1)
        self.assertEqual(tt_zkml.pot_k(0.0, 8), 127)
        self.assertEqual(tt_zkml.pot_k(1.0, 16), 14)

    def test_errors(self):
        with self.assertRaises(tt_zkml.TtZkmlError) as c:
            tt_zkml.pot_k(1.0, 12)
        self.assertEqual(c.exception.status, tt_zkml.TTZK_ERR_INVALID_ARGUMENT)

    def test_device_count(self):
        self.assertGreaterEqual(tt_zkml.num_devices(), 0)


class OnDevice(unittest.TestCase):
    """Needs a Tenstorrent device. One device for the whole class, opened and closed on the main thread."""

    @classmethod
    def setUpClass(cls):
        cls.dev = tt_zkml.Device(0)

    @classmethod
    def tearDownClass(cls):
        cls.dev.close()

    def test_forward_against_the_reference(self):
        m, k, n = 32, 64, 32
        rng = random.Random(1)
        x = [rng.uniform(-2, 2) for _ in range(m * k)]
        w = [rng.uniform(-1, 1) for _ in range(k * n)]
        with tt_zkml.Model(self.dev, [tt_zkml.Layer(k, n)], dtype="f32") as model:
            model.load_parameters([w])
            t = model.forward(x, m)
        sx, sw, so, acc, out = reference_layer(x, w, m, k, n)
        self.assertEqual(t.scaler_x, sx)
        self.assertEqual(t.layers[0].scaler_w, sw)
        self.assertEqual(t.layers[0].scaler_out, so)
        self.assertEqual(t.layers[0].acc_values(), acc)
        self.assertEqual(t.output, out)

    def test_two_layers(self):
        rng = random.Random(2)
        w0 = [rng.uniform(-1, 1) for _ in range(64 * 128)]
        w1 = [rng.uniform(-1, 1) for _ in range(128 * 32)]
        x = [rng.uniform(-2, 2) for _ in range(64 * 64)]
        with tt_zkml.Model(self.dev, [tt_zkml.Layer(64, 128, relu=True), tt_zkml.Layer(128, 32)], dtype="bf16") as model:
            model.load_parameters([w0, w1])
            t = model.forward(x, 64)
            again = model.forward(x, 64)
            with self.assertRaises(tt_zkml.TtZkmlError):
                model.forward(x, 16)  # not a multiple of 32
        self.assertEqual(len(t.layers), 2)
        self.assertEqual(len(t.output), 64 * 32)
        self.assertEqual(len(t.acc(0)), 64 * 128)
        self.assertEqual(t.output, again.output)  # deterministic

    def test_bad_chain(self):
        with self.assertRaises(tt_zkml.TtZkmlError):
            tt_zkml.Model(self.dev, [tt_zkml.Layer(64, 128), tt_zkml.Layer(64, 32)])


if __name__ == "__main__":
    unittest.main()
