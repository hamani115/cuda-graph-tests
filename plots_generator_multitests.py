import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plot_w = 9
plot_h = 6
font_size = 10

GPU_COLORS = {
    "2Nvidia-T4": "red",
    "2Nvidia-L4": "darkorange",
    "2AMD-Radeon-Pro-W7800": "blue",
    # "T4": "red",
    # "L4": "darkorange",
    # "AMD": "blue",
}

def transform_string(input_str, split_char, join_char):
    # Split the string by the given character
    parts = input_str.split(split_char)
    # Capitalize the first letter of each part
    capitalized_parts = [part.capitalize() for part in parts]
    # Join them with the new character
    result = join_char.join(capitalized_parts)
    return result

###############################################################################
#                    1) HELPER: EXTRACT TEST NAME + GPU LABEL                #
###############################################################################
def parse_filename(csv_path):
    """
    Returns (test_name, gpu_label) from a filename.
    
    E.g.  'complex_3_different_kernels_T4.csv' 
      ->  test_name='complex_3_different_kernels'
          gpu_label='T4'
    """
    base = os.path.basename(csv_path)
    # Remove extension
    base_no_ext = os.path.splitext(base)[0]
    # You might define your own logic for how to separate “test” from “GPU”.
    # For example, if the GPU label is always the *last* underscore part:
    parts = base_no_ext.split('_')
    if len(parts) > 1:
        gpu_label = parts[-1]  # e.g. T4 or AMD
        test_name = "_".join(parts[:-1])  # everything except last
    else:
        # fallback
        gpu_label = "UNKNOWN"
        test_name = base_no_ext
    return test_name, gpu_label


###############################################################################
#              2) COMBINED PLOT FUNCTIONS: ONE PLOT WITH MULTIPLE LINES       #
###############################################################################
def generate_gputimeperstep_plot_for_test(test_name, group_csvs, output_dir, num_runs):
    """
    For the given test_name, we have a list of (csv_path, gpu_label).
    We'll create one figure, and plot one line per CSV (per GPU).
    """
    plt.figure(figsize=(plot_w, plot_h))

    for csv_path, gpu_label in group_csvs:
        try:
            df = pd.read_csv(csv_path).sort_values('NSTEP')
            nsteps = df['NSTEP'].values

            # Time WITHOUT Graph
            # data_cols_without = ['noneGraphTotalTimeWithout1', 'noneGraphTotalTimeWithout2',
            #                    'noneGraphTotalTimeWithout3', 'noneGraphTotalTimeWithout4']
            data_cols_without = [f"noneGraphTotalTimeWithout{i}" for i in range(1, num_runs+1)]

            mean_without = df[data_cols_without].mean(axis=1).values
            std_without = df[data_cols_without].std(axis=1).values
            time_perstep_without = mean_without / nsteps
            time_perstep_std_without = std_without / nsteps
            

            # Time WITH Graph
            # data_cols_with = ['GraphTotalTimeWithout1', 'GraphTotalTimeWithout2',
            #                 'GraphTotalTimeWithout3', 'GraphTotalTimeWithout4']
            data_cols_with = [f"GraphTotalTimeWithout{i}" for i in range(1, num_runs+1)]

            mean_with = df[data_cols_with].mean(axis=1).values
            std_with = df[data_cols_with].std(axis=1).values
            time_perstep_with = mean_with / nsteps
            time_perstep_std_with = std_with / nsteps

            color = GPU_COLORS.get(gpu_label, "black")  # fallback is black if not found
            
            line_obj1 = plt.errorbar(
                nsteps,
                time_perstep_without,
                yerr=time_perstep_std_without,
                marker='o',
                linestyle='--',
                capsize=3,   # Add little caps on the error bars
                color=color,
                label=f"{gpu_label} - Without Graph"
            )

            line_obj2 = plt.errorbar(
                nsteps,
                time_perstep_with,
                yerr=time_perstep_std_with,
                marker='o',
                linestyle='-',
                capsize=3,
                color=color,
                label=f"{gpu_label} - With Graph*"
            )
            
            
            # Plot lines
            # line_obj1, = plt.plot(nsteps, time_perstep_without, marker='o', linestyle='--',
            #          color=color,
            #          label=f"{gpu_label} - Without Graph")
            
             # Annotate
            if test_name == "complex_3_different_kernels":
                for x, y in zip(nsteps, time_perstep_without):
                    plt.text(x, y+0.00, f"{y:.3f}", fontsize=9, 
                             color=line_obj1[0].get_color(),  # use the line color
                             ha="right", va="top")
            else:
                for x, y in zip(nsteps, time_perstep_without):
                    plt.text(x, y+0.00, f"{y:.3f}*", fontsize=9, 
                            color=line_obj1[0].get_color(),  # use the line color
                            ha="left", va="bottom")

            # line_obj2, = plt.plot(nsteps, time_perstep_with, marker='o', linestyle='-',
            #          color=color,
            #          label=f"{gpu_label} - With Graph+")
            
             # Annotate
            if test_name == "complex_3_different_kernels":
                for x, y in zip(nsteps, time_perstep_with):
                    plt.text(x, y-0.00, f"{y:.3f}+", fontsize=9, 
                            color=line_obj2[0].get_color(),  # use the line color 
                            ha="left", va="bottom")
            else:
                for x, y in zip(nsteps, time_perstep_with):
                    plt.text(x, y-0.00, f"{y:.3f}+", fontsize=9, 
                            color=line_obj2[0].get_color(),  # use the line color 
                            ha="right", va="top")

            
        except Exception as e:
            print(f"Failed to process {csv_path} ({gpu_label}): {e}")

    plt.title(f"[GPU Time/Step] {test_name}", fontsize=font_size)
    plt.xlabel("NSTEP (Number of Iterations)")
    plt.ylabel("Time Per Iteration (ms)")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    outname = f"{test_name}_gputimeperstep.jpg"
    output_path = os.path.join(output_dir, outname)
    plt.savefig(output_path, format='jpg', dpi=300)
    print(f"Saved GPU time/step plot for test [{test_name}] to {output_path}")
    plt.close()

def generate_gpudiffperstep_plot_for_test(test_name, group_csvs, output_dir, num_runs):
    """
    Same pattern for GPU Diff/Step. One line per GPU label.
    """
    plt.figure(figsize=(plot_w, plot_h))

    for csv_path, gpu_label in group_csvs:
        try:
            df = pd.read_csv(csv_path).sort_values('NSTEP')
            nsteps = df['NSTEP'].values

            # data_cols_without = ['DiffPerStepWithout1', 'DiffPerStepWithout2',
            #                    'DiffPerStepWithout3', 'DiffPerStepWithout4']
            data_cols_without = [f"DiffPerStepWithout{i}" for i in range(1, num_runs+1)]

            mean_without = df[data_cols_without].mean(axis=1).values
            std_without = df[data_cols_without].std(axis=1).values
                        
            color = GPU_COLORS.get(gpu_label, "black")  # fallback is black if not found
                        
            line_obj = plt.errorbar(
                nsteps,
                mean_without,
                yerr=std_without,
                marker='o',
                linestyle='-',
                capsize=3,
                color=color,
                label=f"{gpu_label} - Δ(With vs Without)"
            )
            # plt.plot(nsteps, mean_without, marker='o', linestyle='-',
            #          label=f"{gpu_label} - Δ(With vs Without)")
            
            if test_name == "complex_3_different_kernels" and gpu_label == "T4":
                for x, y in zip(nsteps, mean_without):
                    plt.text(x, y, f"{y:.4f}", fontsize=9, 
                                color=line_obj[0].get_color(), 
                                ha="right", va="bottom")
            elif test_name == "complex_3_different_kernels" and gpu_label == "L4":
                for x, y in zip(nsteps, mean_without):
                    plt.text(x, y, f"{y:.4f}", fontsize=9, 
                                color=line_obj[0].get_color(), 
                                ha="left", va="bottom")
            else:
                for x, y in zip(nsteps, mean_without):
                    plt.text(x, y, f"{y:.4f}", fontsize=9,
                            color=line_obj[0].get_color(), 
                            ha="left", va="top")
                
        except Exception as e:
            print(f"Failed to process {csv_path} ({gpu_label}): {e}")

    plt.title(f"[GPU Diff/Step] {test_name}", fontsize=font_size)
    plt.xlabel("NSTEP (Number of Iterations)")
    plt.ylabel("Time Difference Per Iteration (ms)")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    outname = f"{test_name}_gpudiffperstep.jpg"
    output_path = os.path.join(output_dir, outname)
    plt.savefig(output_path, format='jpg', dpi=300)
    print(f"Saved GPU diff/step plot for test [{test_name}] to {output_path}")
    plt.close()

def generate_gpudiffpercent_plot_for_test(test_name, group_csvs, output_dir, num_runs):
    """
    GPU difference percent, one line per GPU label.
    """
    plt.figure(figsize=(plot_w, plot_h))

    for csv_path, gpu_label in group_csvs:
        try:
            df = pd.read_csv(csv_path).sort_values('NSTEP')
            nsteps = df['NSTEP'].values

            # data_cols_without = ['DiffPercentWithout1', 'DiffPercentWithout2',
            #                    'DiffPercentWithout3', 'DiffPercentWithout4']
            data_cols_without = [f"DiffPercentWithout{i}" for i in range(1, num_runs+1)]

            mean_without = df[data_cols_without].mean(axis=1).values
            std_without = df[data_cols_without].std(axis=1).values
            
            color = GPU_COLORS.get(gpu_label, "black")  # fallback is black if not found
            
            line_obj = plt.errorbar(
                nsteps,
                mean_without,
                yerr=std_without,
                marker='o',
                linestyle='-',
                capsize=3,
                color=color,
                label=f"{gpu_label} - %Diff(With vs Without)"
            )
            
            # plt.plot(nsteps, mean_without, marker='o', linestyle='-',
            #          label=f"{gpu_label} - %Diff(With vs Without)")
            if test_name == "complex_3_different_kernels" and gpu_label == "T4":
                for x, y in zip(nsteps, mean_without):
                    plt.text(x, y, f"{y:.2f}%", fontsize=9, 
                             color=line_obj[0].get_color(), 
                             ha="right", va="bottom")
            else:
                for x, y in zip(nsteps, mean_without):
                    plt.text(x, y, f"{y:.2f}%", fontsize=9, 
                            color=line_obj[0].get_color(), 
                            ha="left", va="top")
            
        except Exception as e:
            print(f"Failed to process {csv_path} ({gpu_label}): {e}")

    plt.title(f"[GPU Diff%] {test_name}", fontsize=font_size)
    plt.xlabel("NSTEP (Number of Iterations)")
    plt.ylabel("Total Time Difference (%)")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    outname = f"{test_name}_gpudiffpercent.jpg"
    output_path = os.path.join(output_dir, outname)
    plt.savefig(output_path, format='jpg', dpi=300)
    print(f"Saved GPU diff% plot for test [{test_name}] to {output_path}")
    plt.close()


###############################################################################
#                 (Similarly) for CPU time, Launch time, etc.                 #
###############################################################################
def generate_cputimeperstep_plot_for_test(test_name, group_csvs, output_dir, num_runs):
    plt.figure(figsize=(plot_w, plot_h))
    for csv_path, gpu_label in group_csvs:
        try:
            df = pd.read_csv(csv_path).sort_values('NSTEP')
            nsteps = df['NSTEP'].values

            # data_cols_without = ['ChronoNoneGraphTotalTimeWithout1', 'ChronoNoneGraphTotalTimeWithout2',
            #                    'ChronoNoneGraphTotalTimeWithout3', 'ChronoNoneGraphTotalTimeWithout4']
            data_cols_without = [f"ChronoNoneGraphTotalTimeWithout{i}" for i in range(1, num_runs+1)]

            mean_without = df[data_cols_without].mean(axis=1).values
            std_without = df[data_cols_without].std(axis=1).values
            time_perstep_without = mean_without / nsteps
            time_perstep_std_without = std_without / nsteps

            # data_cols_with = ['ChronoGraphTotalTimeWithout1', 'ChronoGraphTotalTimeWithout2',
            #                 'ChronoGraphTotalTimeWithout3', 'ChronoGraphTotalTimeWithout4']
            data_cols_with = [f"ChronoGraphTotalTimeWithout{i}" for i in range(1, num_runs+1)]

            mean_with = df[data_cols_with].mean(axis=1).values
            std_with = df[data_cols_with].std(axis=1).values
            time_perstep_with = mean_with / nsteps
            time_perstep_std_with = std_with / nsteps

            color = GPU_COLORS.get(gpu_label, "black")  # fallback is black if not found
            
            line_obj1 = plt.errorbar(
                nsteps,
                time_perstep_without,
                yerr=time_perstep_std_without,
                marker='o',
                linestyle='--',
                capsize=3,   # Add little caps on the error bars
                color=color,
                label=f"{gpu_label} - Without Graph"
            )

            line_obj2 = plt.errorbar(
                nsteps,
                time_perstep_with,
                yerr=time_perstep_std_with,
                marker='o',
                linestyle='-',
                capsize=3,
                color=color,
                label=f"{gpu_label} - With Graph*"
            )
            
            # line_obj1, = plt.plot(nsteps, time_perstep_without, marker='o', linestyle='--',
            #          color=color,
            #          label=f"{gpu_label} - Without Graph")
            for x, y in zip(nsteps, time_perstep_without):
                plt.text(x, y+0.01, f"{y:.2f}", fontsize=9, 
                         color=line_obj1[0].get_color(),  # use the line color 
                         ha="left", va="bottom")
            # line_obj2, = plt.plot(nsteps, time_perstep_with, marker='o', linestyle='-',
            #          color=color,
            #          label=f"{gpu_label} - With Graph+")
            for x, y in zip(nsteps, time_perstep_with):
                plt.text(x, y-0.01, f"{y:.2f}*", fontsize=9, 
                         color=line_obj2[0].get_color(),  # use the line color
                         ha="right", va="top")

        except Exception as e:
            print(f"Failed to process {csv_path} ({gpu_label}): {e}")

    plt.title(f"[CPU Time/Step] {test_name}", fontsize=font_size)
    plt.xlabel("NSTEP (Number of Iterations)")
    plt.ylabel("Time Per Iteration (ms)")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    outname = f"{test_name}_cputimeperstep.jpg"
    output_path = os.path.join(output_dir, outname)
    plt.savefig(output_path, format='jpg', dpi=300)
    print(f"Saved CPU time/step plot for test [{test_name}] to {output_path}")
    plt.close()

def generate_cpudiffperstep_plot_for_test(test_name, group_csvs, output_dir, num_runs):
    plt.figure(figsize=(plot_w, plot_h))
    for csv_path, gpu_label in group_csvs:
        try:
            df = pd.read_csv(csv_path).sort_values('NSTEP')
            nsteps = df['NSTEP'].values

            # data_cols_without = ['ChronoDiffPerStepWithout1','ChronoDiffPerStepWithout2',
            #                    'ChronoDiffPerStepWithout3','ChronoDiffPerStepWithout4']
            data_cols_without = [f"ChronoDiffPerStepWithout{i}" for i in range(1, num_runs+1)]

            mean_without = df[data_cols_without].mean(axis=1).values
            std_without = df[data_cols_without].std(axis=1).values
            
            color = GPU_COLORS.get(gpu_label, "black")  # fallback is black if not found
            
            line_obj = plt.errorbar(
                nsteps,
                mean_without,
                yerr=std_without,
                marker='o',
                linestyle='-',
                capsize=3,
                color=color,
                label=f"{gpu_label} - Δ(With vs Without)"
            )
            
            # plt.plot(nsteps, mean_without, marker='o', linestyle='-',
            #          label=f"{gpu_label} - Δ(With vs Without)")
            for x, y in zip(nsteps, mean_without):
                plt.text(x, y, f"{y:.4f}", fontsize=9, 
                         color=line_obj[0].get_color(),
                         ha="left", va="top")
            

        except Exception as e:
            print(f"Failed to process {csv_path} ({gpu_label}): {e}")

    plt.title(f"[CPU Diff/Step] {test_name}", fontsize=font_size)
    plt.xlabel("NSTEP (Number of Iterations)")
    plt.ylabel("Time Difference Per Iteration (ms)")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    outname = f"{test_name}_cpudiffperstep.jpg"
    output_path = os.path.join(output_dir, outname)
    plt.savefig(output_path, format='jpg', dpi=300)
    print(f"Saved CPU diff/step plot for test [{test_name}] to {output_path}")
    plt.close()

def generate_cpudiffpercent_plot_for_test(test_name, group_csvs, output_dir, num_runs):
    plt.figure(figsize=(plot_w, plot_h))
    for csv_path, gpu_label in group_csvs:
        try:
            df = pd.read_csv(csv_path).sort_values('NSTEP')
            nsteps = df['NSTEP'].values
            # data_cols_without = ['ChronoDiffPercentWithout1','ChronoDiffPercentWithout2',
            #                    'ChronoDiffPercentWithout3','ChronoDiffPercentWithout4']
            data_cols_without = [f"ChronoDiffPercentWithout{i}" for i in range(1, num_runs+1)]

            mean_without = df[data_cols_without].mean(axis=1).values
            std_without = df[data_cols_without].std(axis=1).values
            
            color = GPU_COLORS.get(gpu_label, "black")  # fallback is black if not found
            
            line_obj = plt.errorbar(
                nsteps,
                mean_without,
                yerr=std_without,
                marker='o',
                linestyle='-',
                capsize=3,
                color=color,
                label=f"{gpu_label} - %Diff(With vs Without)"
            )
            
            # plt.plot(nsteps, data_without, marker='o', linestyle='-',
            #          label=f"{gpu_label} - %Diff(With vs Without)")
            for x, y in zip(nsteps, mean_without):
                plt.text(x, y, f"{y:.2f}%", fontsize=9, 
                         color=line_obj[0].get_color(),
                         ha="left", va="top")
                
        except Exception as e:
            print(f"Failed to process {csv_path} ({gpu_label}): {e}")

    plt.title(f"[CPU Diff%] {test_name}", fontsize=font_size)
    plt.xlabel("NSTEP (Number of Iterations)")
    plt.ylabel("Total Time Difference (%)")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    outname = f"{test_name}_cpudiffpercent.jpg"
    output_path = os.path.join(output_dir, outname)
    plt.savefig(output_path, format='jpg', dpi=300)
    print(f"Saved CPU diff% plot for test [{test_name}] to {output_path}")
    plt.close()


###############################################################################
#     (Similarly) for Launch time/time_perstep_withstep, Launch diff, etc.    #
###############################################################################
def generate_launchtimeperstep_plot_for_test(test_name, group_csvs, output_dir, num_runs):
    plt.figure(figsize=(plot_w, plot_h))
    for csv_path, gpu_label in group_csvs:
        try:
            df = pd.read_csv(csv_path).sort_values('NSTEP')
            nsteps = df['NSTEP'].values

            # data_cols_without = ['ChronoNoneGraphTotalLaunchTimeWithout1','ChronoNoneGraphTotalLaunchTimeWithout2',
            #                    'ChronoNoneGraphTotalLaunchTimeWithout3','ChronoNoneGraphTotalLaunchTimeWithout4']
            data_cols_without = [f"ChronoNoneGraphTotalLaunchTimeWithout{i}" for i in range(1, num_runs+1)]

            mean_without = df[data_cols_without].mean(axis=1).values
            std_without = df[data_cols_without].std(axis=1).values
            time_perstep_without = (mean_without / nsteps) * 1000
            time_perstep_std_without = (std_without / nsteps) * 1000

            # data_cols_with = ['ChronoGraphTotalLaunchTimeWithout1','ChronoGraphTotalLaunchTimeWithout2',
            #                 'ChronoGraphTotalLaunchTimeWithout3','ChronoGraphTotalLaunchTimeWithout4']
            data_cols_with = [f"ChronoGraphTotalLaunchTimeWithout{i}" for i in range(1, num_runs+1)]

            mean_with = df[data_cols_with].mean(axis=1).values
            std_with = df[data_cols_with].std(axis=1).values
            time_perstep_with = (mean_with / nsteps) * 1000
            time_perstep_std_with = (std_with / nsteps) * 1000

            color = GPU_COLORS.get(gpu_label, "black")  # fallback is black if not found
            
            line_obj1 = plt.errorbar(
                nsteps,
                time_perstep_without,
                yerr=time_perstep_std_without,
                marker='o',
                linestyle='--',
                capsize=3,   # Add little caps on the error bars
                color=color,
                label=f"{gpu_label} - Without Graph"
            )

            line_obj2 = plt.errorbar(
                nsteps,
                time_perstep_with,
                yerr=time_perstep_std_with,
                marker='o',
                linestyle='-',
                capsize=3,
                color=color,
                label=f"{gpu_label} - With Graph*"
            )
            
            # line_obj1, = plt.plot(nsteps, time_perstep_without, marker='o', linestyle='--',
            #          color=color,
            #          label=f"{gpu_label} - Without Graph")
            for x, y in zip(nsteps, time_perstep_without):
                plt.text(x, y+0.0000, f"{y:.1f}", fontsize=9, 
                         color=line_obj1[0].get_color(),  # use the line color
                         ha="left", va="bottom")
            # line_obj2, = plt.plot(nsteps, time_perstep_with, marker='o', linestyle='-',
            #          color=color,
            #          label=f"{gpu_label} - With Graph+")
            for x, y in zip(nsteps, time_perstep_with):
                plt.text(x, y-0.0000, f"{y:.1f}*", fontsize=9, 
                         color=line_obj2[0].get_color(),  # use the line color 
                         ha="left", va="top")
        except Exception as e:
            print(f"Failed to process {csv_path} ({gpu_label}): {e}")

    plt.title(f"[Launch Time/Step] {test_name}", fontsize=font_size)
    plt.xlabel("NSTEP (Number of Iterations)")
    plt.ylabel("Time Per Iteration (μs)")
    # plt.ylabel("Time Per Iteration (ms)")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    outname = f"{test_name}_launchtimeperstep.jpg"
    output_path = os.path.join(output_dir, outname)
    plt.savefig(output_path, format='jpg', dpi=300)
    print(f"Saved Launch time/step plot for test [{test_name}] to {output_path}")
    plt.close()

def generate_launchdiffperstep_plot_for_test(test_name, group_csvs, output_dir, num_runs):
    plt.figure(figsize=(plot_w, plot_h))
    for csv_path, gpu_label in group_csvs:
        try:
            df = pd.read_csv(csv_path).sort_values('NSTEP')
            nsteps = df['NSTEP'].values

            # data_cols_without = [
            #     'ChronoDiffLaunchTimeWithout1',
            #     'ChronoDiffLaunchTimeWithout2',
            #     'ChronoDiffLaunchTimeWithout3',
            #     'ChronoDiffLaunchTimeWithout4'
            # ]
            data_cols_without = [f"ChronoDiffLaunchTimeWithout{i}" for i in range(1, num_runs+1)]

            
            mean_without = (df[data_cols_without].mean(axis=1).values)
            std_without = (df[data_cols_without].std(axis=1).values)
            time_perstep_without = (mean_without / nsteps) * 1000
            time_perstep_std_without = (std_without / nsteps) * 1000
            
            
            color = GPU_COLORS.get(gpu_label, "black")  # fallback is black if not found
                        
            line_obj = plt.errorbar(
                nsteps,
                time_perstep_without,
                yerr=time_perstep_std_without,
                marker='o',
                linestyle='-',
                capsize=3,
                color=color,
                label=f"{gpu_label} - Δ(With vs Without)"
            )

            # plt.plot(nsteps, mean_without, marker='o', linestyle='-',
            #          label=f"{gpu_label} - Δ(With vs Without)")
            for x, y in zip(nsteps, time_perstep_without):
                plt.text(x, y, f"{y:.2f}", fontsize=9, 
                         color=line_obj[0].get_color(),
                         ha="left", va="top")
        except Exception as e:
            print(f"Failed to process {csv_path} ({gpu_label}): {e}")

    plt.title(f"[Launch Diff/Step] {test_name}", fontsize=font_size)
    plt.xlabel("NSTEP (Number of Iterations)")
    plt.ylabel("Time Difference (μs)")
    # plt.ylabel("Time Difference (ms)")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    outname = f"{test_name}_launchdiffperstep.jpg"
    output_path = os.path.join(output_dir, outname)
    plt.savefig(output_path, format='jpg', dpi=300)
    print(f"Saved Launch diff/step plot for test [{test_name}] to {output_path}")
    plt.close()

def generate_launchdiffpercent_plot_for_test(test_name, group_csvs, output_dir, num_runs):
    plt.figure(figsize=(plot_w, plot_h))
    for csv_path, gpu_label in group_csvs:
        try:
            df = pd.read_csv(csv_path).sort_values('NSTEP')
            nsteps = df['NSTEP'].values

            # data_cols_without = ['ChronoDiffLaunchPercentWithout1','ChronoDiffLaunchPercentWithout2',
            #                    'ChronoDiffLaunchPercentWithout3','ChronoDiffLaunchPercentWithout4']
            data_cols_without = [f"ChronoDiffLaunchPercentWithout{i}" for i in range(1, num_runs+1)]

            mean_without = df[data_cols_without].mean(axis=1).values
            std_without = df[data_cols_without].std(axis=1).values
            
            color = GPU_COLORS.get(gpu_label, "black")  # fallback is black if not found
            
            line_obj = plt.errorbar(
                nsteps,
                mean_without,
                yerr=std_without,
                marker='o',
                linestyle='-',
                capsize=3,
                color=color,
                label=f"{gpu_label} - %Diff(With vs Without)"
            )
            
            # plt.plot(nsteps, mean_without, marker='o', linestyle='-',
            #          label=f"{gpu_label} - %Diff(With vs Without)")
            for x, y in zip(nsteps, mean_without):
                plt.text(x, y, f"{y:.1f}%", fontsize=9, 
                         color=line_obj[0].get_color(),
                         ha="left", va="top")
        except Exception as e:
            print(f"Failed to process {csv_path} ({gpu_label}): {e}")

    plt.title(f"[Launch Diff%] {test_name}", fontsize=font_size)
    plt.xlabel("NSTEP (Number of Iterations)")
    plt.ylabel("Total Time Difference (%)")
    plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    outname = f"{test_name}_launchdiffpercent.jpg"
    output_path = os.path.join(output_dir, outname)
    plt.savefig(output_path, format='jpg', dpi=300)
    print(f"Saved Launch diff% plot for test [{test_name}] to {output_path}")
    plt.close()


###############################################################################
#                             MAIN FUNCTION                                   #
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description='Generate combined GPU/CPU/Launch plots from multiple CSV files.'
    )
    parser.add_argument(
        'csv_files', 
        metavar='CSV', 
        type=str, 
        nargs='+',
        help='Path(s) to the input CSV file(s).'
    )
    parser.add_argument(
        '-o', '--output', 
        type=str, 
        default='plots',
        help='Directory to save the output plots. Defaults to "./plots".'
    )
    parser.add_argument(
        '--num_runs',
        type=int,
        default=4,
        help='Number of runs for each measurement column (default: 4).'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # 1) Group CSVs by test_name. For each CSV, parse out test_name + GPU label
    test_groups = {}  # dict: test_name -> list of (csv_path, gpu_label)
    for csv_path in args.csv_files:
        if not os.path.isfile(csv_path):
            print(f"File not found: {csv_path}")
            continue

        test_name, gpu_label = parse_filename(csv_path)
        if test_name not in test_groups:
            test_groups[test_name] = []
        test_groups[test_name].append((csv_path, gpu_label))

    # 2) For each test_name group, generate the various plots.
    #    Each plot function will create a single figure with one line per GPU label.
    for test_name, group_csvs in test_groups.items():
        # GPU
        generate_gputimeperstep_plot_for_test(test_name, group_csvs, output_dir, args.num_runs)
        generate_gpudiffperstep_plot_for_test(test_name, group_csvs, output_dir, args.num_runs)
        generate_gpudiffpercent_plot_for_test(test_name, group_csvs, output_dir, args.num_runs)
        # CPU
        generate_cputimeperstep_plot_for_test(test_name, group_csvs, output_dir, args.num_runs)
        generate_cpudiffperstep_plot_for_test(test_name, group_csvs, output_dir, args.num_runs)
        generate_cpudiffpercent_plot_for_test(test_name, group_csvs, output_dir, args.num_runs)
        # LAUNCH
        generate_launchtimeperstep_plot_for_test(test_name, group_csvs, output_dir, args.num_runs)
        generate_launchdiffperstep_plot_for_test(test_name, group_csvs, output_dir, args.num_runs)
        generate_launchdiffpercent_plot_for_test(test_name, group_csvs, output_dir, args.num_runs)

        # Extras:
        # generate_cputotaltime_plot_for_test(...)
        # generate_launchtotaltime_plot_for_test(...)
        # generate_launchdifftotal_plot_for_test(...)

if __name__ == "__main__":
    main()
