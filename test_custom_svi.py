import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from bnn.bnn_utils import PerDatapointTraceELBO

# Example model
def model(x_data, y_data):
    # Sample global mean and sd
    mean = pyro.sample("mean", pyro.distributions.Normal(0, 1))
    sd = pyro.sample("sd", pyro.distributions.LogNormal(0, 1))

    # Expand to match y_data shape
    mean = mean.expand(y_data.shape)
    sd = sd.expand(y_data.shape)

    with pyro.plate("data", x_data.size(0)):
        # Observation
        prediction = pyro.sample("obs", pyro.distributions.Normal(mean, sd).to_event(1), obs=y_data)

# Example guide
def guide(x_data, y_data):
    mean_param = pyro.param("mean_param", torch.tensor(0.0))
    sd_param = pyro.param("sd_param", torch.tensor(1.0), constraint=pyro.distributions.constraints.positive)
    pyro.sample("mean", pyro.distributions.Normal(mean_param, sd_param))
    pyro.sample("sd", pyro.distributions.LogNormal(mean_param, sd_param))

# Generate synthetic data
torch.manual_seed(0)
batch_size = 5
x_data = torch.arange(0, batch_size, dtype=torch.float32).unsqueeze(1)
y_data = 2 * x_data + torch.randn_like(x_data) * 0.5

# Pyro optimizers
optimizer = Adam({"lr": 0.01})

# Standard Trace_ELBO with svi.step()
svi_standard = SVI(model, guide, optimizer, loss=Trace_ELBO())
loss_standard = svi_standard.step(x_data, y_data)

# Custom implementation with PerDatapointTraceELBO
elbo_custom = PerDatapointTraceELBO(num_particles=1, vectorize_particles=True)
svi_custom = SVI(model, guide, optimizer, loss=elbo_custom)

# Custom loss and individual datapoint losses
total_loss_custom, per_datapoint_losses = elbo_custom.loss_and_grads(model, guide, x_data, y_data)

# Compare results
print(f"Standard ELBO Loss: {loss_standard}")
print(f"Custom Total Loss: {total_loss_custom}")
print(f"Per-Datapoint Losses: {per_datapoint_losses}")
print(f"Sum of Per-Datapoint Losses: {sum(per_datapoint_losses)}")

# Assertions
assert abs(loss_standard - total_loss_custom) < 1e-5, \
    f"Total loss mismatch: Standard {loss_standard} vs Custom {total_loss_custom}"
assert abs(sum(per_datapoint_losses) - total_loss_custom) < 1e-5, \
    f"Sum of per-datapoint losses mismatch: {sum(per_datapoint_losses)} vs Total {total_loss_custom}"

# Print parameter updates
print("\nParameter values:")
for name, param in pyro.get_param_store().items():
    print(f"{name}: {param.item()}")

# Verify gradients
for name, param in pyro.get_param_store().items():
    print(f"{name} gradient: {param.grad}")
    assert param.grad is not None, f"No gradient for parameter {name}"

# Perform additional steps with custom implementation
print("\nAdditional steps with custom implementation:")
for step in range(3):
    total_loss_custom, per_datapoint_losses = elbo_custom.loss_and_grads(model, guide, x_data, y_data)
    print(f"Step {step + 1}: Total Loss = {total_loss_custom}, Per-DataPoint Losses = {per_datapoint_losses}")

