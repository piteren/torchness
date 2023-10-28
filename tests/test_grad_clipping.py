import torch
import unittest

from torchness.grad_clipping import clip_grad_norm_, GradClipperMAVG


class TestGradClipping(unittest.TestCase):

    def test_clip(self):

        vec = torch.rand(100)
        grad = torch.rand(100)
        vec.grad = grad

        print(grad.norm())

        cn = clip_grad_norm_(vec, 1.5)
        print(cn)
        print(grad.norm())

        self.assertTrue(1.49 < float(grad.norm()) < 1.51)

    def test_GradClipperMAVG(self):

        class Mod(torch.nn.Module):

            def __init__(self, params):
                super().__init__()
                self.params = params

            def parameters(self, recurse: bool=True):
                return self.params

        vec = torch.rand(100)
        grad = torch.rand(100)
        vec.grad = grad

        clip_grad_norm_(vec, 0.5)

        grad_1 = torch.clone(vec.grad)

        gc = GradClipperMAVG(
            module=     Mod(vec),
            factor=     0.1,
            first_avg=  True,
            start_val=  0.5,
            max_clip=   1.0,
            max_upd=    1.5)

        f = 1
        for ix in range(100):
            gres = gc.clip()
            print(f'{ix:02} {gres["gg_norm"]:.5f} {gres["gg_norm_clip"]:.5f} {vec.grad.norm():.5f}')
            if ix > 10: f = 4
            if ix > 50: f = 1
            vec.grad = grad_1 * f
        print()

        gc = GradClipperMAVG(
            module=     Mod(vec),
            factor=     0.1,
            first_avg=  False,
            start_val=  0.1,
            max_upd=    1.5,
            #loglevel=   10,
        )

        f = 1
        for ix in range(100):
            gres = gc.clip()
            print(f'{ix:02} {gres["gg_norm"]:.5f} {gres["gg_norm_clip"]:.5f} {vec.grad.norm():.5f}')
            if ix > 10: f = 4
            if ix > 50: f = 1
            vec.grad = grad_1 * f