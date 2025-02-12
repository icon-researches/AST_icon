import argparse
import random
import os
from time import time as t

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from keras.datasets import mnist
from sklearn.preprocessing import minmax_scale

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import assign_labels, all_activity, proportion_weighting
from bindsnet.memstdp.MemSTDP_models import DiehlAndCook2015_MemSTDP
from bindsnet.memstdp.MemSTDP_learning import PostPre, Constraint_PostPre
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.memstdp.plotting_weights_counts import hist_weights

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=400)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--weight_scale", type=float, default=1.0)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=3)
parser.add_argument("--ST", type=bool, default=False)
parser.add_argument("--AST", type=bool, default=False)
parser.add_argument("--drop_num", type=int, default=320)
parser.add_argument("--reinforce_num", type=int, default=25)
parser.add_argument("--FT", dest="fault_type", default=None)
parser.add_argument("--FS_input_num", type=int, default=0)
parser.add_argument("--FS_exc_num", type=int, default=0)
parser.add_argument("--Pruning", type=bool, default=False)
parser.add_argument("--Noise", type=bool, default=False)
parser.add_argument("--std", type=float, default=0.2)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--spare_gpu", dest="spare_gpu", default=0)
parser.set_defaults(plot=False, gpu=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
weight_scale = args.weight_scale
progress_interval = args.progress_interval
update_interval = args.update_interval
ST = args.ST
AST = args.AST
drop_num = args.drop_num
reinforce_num = args.reinforce_num
FT = args.fault_type
FS_input_num = args.FS_input_num
FS_exc_num = args.FS_exc_num
Pruning = args.Pruning
Noise = args.Noise
std = args.std
plot = args.plot
gpu = args.gpu
spare_gpu = args.spare_gpu

# Sets up Gpu use
if spare_gpu != 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(spare_gpu)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)

print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

print(n_workers, os.cpu_count() - 1)

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
num_inputs = 784

# Build network.
network = DiehlAndCook2015_MemSTDP(
    n_inpt=num_inputs,
    n_neurons=n_neurons,
    update_rule=PostPre,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=num_inputs / 10 * weight_scale,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Load MNIST data.
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../data/MNIST",
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../data/MNIST",
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# AST settings
pre_average = []
drop_input = []
drop_mask = torch.ones_like(torch.zeros(num_inputs, n_neurons)).to(device)
reinforce_input = []
reinforce_ref = []
reinforce_mask = torch.zeros_like(torch.zeros(num_inputs, n_neurons)).to(device)

pre = mnist.load_data()
pre_x = minmax_scale(pre[0][0].reshape(60000, num_inputs))
pre_y = pre[0][1].reshape(60000, 1)

if Noise:
    noise_data = []
    SNR = []
    for sample in pre_x:
        noise = np.random.normal(0, std, sample.shape)
        noise_added = minmax_scale(sample + noise)
        noise_data.append(noise_added)
        SNR.append(20 * np.log10(np.linalg.norm(sample, 1) / np.linalg.norm(noise, 1)))

    pre_x = np.array(noise_data)
    SNR = np.average(np.array(SNR))
    print('SNR =', SNR, 'dB')

    train_dataset = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        "../data/MNIST",
        download=True,
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(lambda x: x + torch.randn(x.size()) * std),
             transforms.Lambda(lambda x: torch.tensor(minmax_scale(x.cpu().numpy().reshape(-1))).
                               reshape(1, 28, 28) * intensity)]
        ),
    )

    test_dataset = MNIST(
        PoissonEncoder(time=time, dt=dt),
        None,
        "../data/MNIST",
        download=True,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(lambda x: x + torch.randn(x.size()) * std),
             transforms.Lambda(lambda x: torch.tensor(minmax_scale(x.cpu().numpy().reshape(-1))).
                               reshape(1, 28, 28) * intensity)]
        ),
    )

preprocessed = np.concatenate((pre_x, pre_y), axis=1)
preprocessed = preprocessed[preprocessed[:, num_inputs].argsort()]
preprocessed = preprocessed[:, 0:num_inputs]

pre_size = 6000
repeat = int(np.ceil(n_neurons / 10))
entire = np.sort(np.mean(preprocessed, axis=0))

if ST:
    for i in range(10):
        pre_average.append(np.mean(preprocessed[i * pre_size:(i + 1) * pre_size], axis=0))

        if AST:
            drop_num = len(np.where(pre_average[i] <= entire[int(num_inputs * 0.3) - 1])[0])
            reinforce_num = len(np.where(pre_average[i] >= entire[int(num_inputs * 1.0) - 1])[0])

        drop_input.append(np.argwhere(pre_average[i] < np.sort(pre_average[i])[0:drop_num + 1][-1]).flatten())
        reinforce_input.append(
            np.argwhere(pre_average[i] > np.sort(pre_average[i])[0:num_inputs - reinforce_num][-1]).flatten())
        if reinforce_num != 0:
            values = np.sort(pre_average[i])[::-1][:reinforce_num]
            reinforce_ref.append(minmax_scale(values, feature_range=(0.9, 1.0)) * weight_scale)
        else:
            reinforce_ref.append([])

    drop_input *= repeat
    reinforce_input *= repeat
    reinforce_ref *= repeat

    for i in range(n_neurons):
        for j in drop_input[i]:
            drop_mask[j][i] = 0

    for i in range(n_neurons):
        for j in reinforce_input[i]:
            reinforce_mask[j][i] = reinforce_ref[i][int(np.where(j == reinforce_input[i])[0])]

# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Ai"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
hist_ax = None
voltage_axes, voltage_ims = None, None

# Dead synapse simulation
if FT == "SA0":
    fault_input = []
    for i in range(FS_exc_num):
        fault_input.append(random.sample(range(0, num_inputs), FS_input_num))
    fault_exc = random.sample(range(0, n_neurons), FS_exc_num)

    fault_mask = torch.ones_like(torch.zeros((num_inputs, n_neurons))).to(device)
    for i in range(len(fault_exc)):
        for j in fault_input[i]:
            fault_mask[j, fault_exc[i]] = 0

elif FT == "SA1":
    fault_input = []
    for i in range(FS_exc_num):
        fault_input.append(random.sample(range(0, num_inputs), FS_input_num))
    fault_exc = random.sample(range(0, n_neurons), FS_exc_num)

    fault_mask = torch.zeros_like(torch.zeros((num_inputs, n_neurons))).to(device)
    for i in range(len(fault_exc)):
        for j in fault_input[i]:
            fault_mask[j, fault_exc[i]] = num_inputs / 10

else:
    fault_mask = torch.ones_like(torch.zeros((num_inputs, n_neurons))).to(device)

# Train the network.
print("\nBegin training.\n")
start = t()
print("check accuracy per", update_interval)
for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        if step > n_train:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if step % update_interval == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record,
                assignments=assignments,
                n_labels=n_classes,

            )

            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        labels.append(batch["label"])

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1, Pruning=Pruning,
                    ST=ST, drop_mask=drop_mask, reinforce_mask=reinforce_mask,
                    fault_type=FT, fault_mask=fault_mask)

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # Add to spikes recording.
        spike_record[step % update_interval] = spikes["Ae"].get("s").squeeze()

        # Optionally plot various simulation information.
        if plot:
            image = batch["image"].view(28, 28)
            inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
            input_exc_weights = network.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
            voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )

            weight_collections = network.connections[("X", "Ae")].w.reshape(-1).tolist()
            hist_ax = hist_weights(weight_collections, ax=hist_ax)

            plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1, Pruning=Pruning,
                ST=ST, drop_mask=drop_mask, reinforce_mask=reinforce_mask,
                fault_type=FT, fault_mask=fault_mask)

    # Add to spikes recording.
    spike_record[0] = spikes["Ae"].get("s").squeeze()

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record,
        assignments=assignments,
        n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test * 100))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test * 100))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")
