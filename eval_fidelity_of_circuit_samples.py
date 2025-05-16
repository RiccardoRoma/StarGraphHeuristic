import os
import tarfile
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import cycle
import matplotlib as mpl

def collect_yaml_files(base_dir):
    """Collect all yaml result files from all subdirectories of base_dir.

    Args:
        base_dir: path/to/base_dir

    Returns:
        list of file paths of the collected yaml files.
    """
    yaml_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_result.yaml') or file.endswith('_result.yml'):
                full_path = os.path.join(root, file)
                yaml_files.append(full_path)
    return yaml_files

def bundle_yaml_files(base_dir, output_tar_path):
    """Combine all yaml result files in all sub-dir of base_dir into a tar.gz file

    Args:
        base_dir: path/to/base_dir
        output_tar_path: filepath of the .tar.gz file.
    """
    yaml_files = collect_yaml_files(base_dir)
    
    with tarfile.open(output_tar_path, "w:gz") as tar:
        for full_path in yaml_files:
            arcname = os.path.basename(full_path)
            tar.add(full_path, arcname=arcname)

    print(f"Archived {len(yaml_files)} YAML file(s) to: {output_tar_path}")

def load_data_from_tar(archive_path):
    """Load the fidelity data and number of qubits from tar archives.

    Args:
        archive_path: path to tar archive

    Returns:
        data dictionary, where the keys correspond to different number of qubits and the values correspond to the fidelities.
    """
    data_by_x = defaultdict(list)

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and (member.name.endswith(".yaml") or member.name.endswith(".yml")):
                f = tar.extractfile(member)
                if f:
                    try:
                        d = yaml.safe_load(f)
                        x = d['num_qubits']
                        fidelity_arr = d['fidelity']
                        y = fidelity_arr[0]
                        std = fidelity_arr[1]
                        data_by_x[x].append((y, std))
                    except Exception as e:
                        print(f"Skipping {member.name}: {e}")
    return data_by_x

def compute_weighted_averages(data_by_x):
    """Compute the average and the standard error of the mean (sem) of a data dictionary. 

    Args:
        data_by_x: data dictionary, where the keys correspond to different x values and the values correspond to the data that should be averaged.

    Returns:
        list of x values, corresponding list of averaged y values and list of sem values
    """
    xs, ys, yerrs = [], [], []
    for x in sorted(data_by_x.keys()):
        values = data_by_x[x]
        ys_x = np.asarray([v[0] for v in values])
        stds_x = np.asarray([v[1] for v in values])

        mean = np.mean(ys_x)
        sem = np.std(ys_x, ddof=1) / np.sqrt(len(ys_x))

        # weights = 1/stds_x**2
        # mean = np.sum(weights * ys_x) / np.sum(weights)
        # sem = np.sqrt(1 / np.sum(weights))
        

        xs.append(x)
        ys.append(mean)
        yerrs.append(sem)

    return np.array(xs), np.array(ys), np.array(yerrs)

def plot_average_fidelities_from_archives(
    *archives,
    x_label="x",
    legend_labels=None,
    figsize=(16, 9),
    title="",
    fname=None
):
    """Plot the average fidelities over the number of qubits from result tar.gz archives. This function averages
    over the fidelities in one archive (considering different number of qubits). Each archive is plotted
    as a different curve. 

    Args:
        archives: list of tar files that should be plotted in one figure.
        x_label: x axis label. Defaults to "x".
        legend_labels: labels of the differnt archives shown in the legend. Defaults to None, which means it will take the file name.
        figsize: size of the figure. Defaults to (16, 9).
        title: title of the figure. Defaults to "".
        fname: filename to save the plot. Defaults to None, which means not saving this plot.
    """
    
    #colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'brown', 'cyan'])
    colors = cycle(mpl.color_sequences["tab10"])
    markers = cycle(["o", "x", "d", "s", "*", "p"])
    linestyles = cycle(['-', '--', '-.', ':'])

    # The parameters for plotting
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif", weight="bold", size="20")
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", size=10)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", size=5)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", size=10)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", size=5)
    plt.rc("ytick.minor", width=2)

    # Marker size
    msize = 10
    lw = 2

    plt.figure(figsize=figsize)

    for idx, archive_path in enumerate(archives):
        label = (
            legend_labels[idx] if legend_labels and idx < len(legend_labels)
            else os.path.basename(archive_path).replace('.tar.gz', '')
        )

        color = next(colors)
        marker = next(markers)
        linestyle = next(linestyles)

        data = load_data_from_tar(archive_path)
        x, y, yerr = compute_weighted_averages(data)

        plt.errorbar(x, y, yerr=yerr, fmt=marker, ms=msize, color=color, label=label, capsize=5)
        plt.plot(x, y, linestyle=linestyle, linewidth=lw, color=color, alpha=0.6)

    plt.xlabel(x_label)
    plt.ylabel("Average Fidelity")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if fname:
        plt.savefig(fname, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bundle YAML files from datapoint folders into a tar.gz archive.")
    parser.add_argument("folder", help="Path to folder A or B")
    parser.add_argument("-o", "--output", default="bundled_yaml.tar.gz", help="Output tar.gz file name (default: bundled_yaml.tar.gz)")
    args = parser.parse_args()
    
    bundle_yaml_files(args.folder, args.output)

    # archives = ["./graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_2/circuit_sampling_fidelity_eval_layout_graph_ibm_brisbane_2_grow_full_noise.tar.gz",
    #             "./graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_2/circuit_sampling_fidelity_eval_layout_graph_ibm_brisbane_2_non-bin_parallel_merge_substate_none_ghz_full_noise.tar.gz"]
# 
    # plot_average_fidelities_from_archives(*archives,
    #                                       x_label="Number of qubits",
    #                                       legend_labels=["state growing", "merge parallel non-binary, size=hd"],
    #                                       figsize=(16, 9),
    #                                       title="",
    #                                       fname="./graph_samples_eval/circuit_sampling_layout_graph_ibm_brisbane_2/circuit_sampling_eval_layout_graph_ibm_brisbane_2_full_noise_fidelity.pdf")

