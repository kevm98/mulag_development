import torch

from ROM_pipeline.runtime import CylinderTargetIntegrator, ROMHistory


def test_target_integrator_enforces_lead_and_limits():
    lower = torch.zeros(1, 4)
    upper = torch.ones(1, 4)
    integrator = CylinderTargetIntegrator(lower, upper, max_target_lead_m=0.02)
    measured = torch.full((1, 4), 0.5)
    integrator.reset(measured)
    target, diagnostics = integrator.step(torch.full((1, 4), 0.1), measured)
    assert torch.allclose(target, torch.full((1, 4), 0.52))
    assert diagnostics["lead_clipped"].all()


def test_history_readiness_and_reset():
    history = ROMHistory(2, 3, 5)
    row = torch.ones(2, 5)
    for _ in range(3):
        history.append(row)
    assert history.ready.all()
    history.reset(torch.tensor([0]))
    assert not history.ready[0] and history.ready[1]
