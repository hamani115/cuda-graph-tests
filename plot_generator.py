import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plot_w = 8
plot_h = 6
font_size = 10

def transform_string(input_str, split_char, join_char):
    # Split the string by the given character
    parts = input_str.split(split_char)
    # Capitalize the first letter of each part
    capitalized_parts = [part.capitalize() for part in parts]
    # Join them with the new character
    result = join_char.join(capitalized_parts)
    return result

# PLOTS EXTRAS
def generate_cputotaltime_plot(csv_path, output_dir, num_runs):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure NSTEP is sorted
        df = df.sort_values('NSTEP')

        # Extract NSTEP values
        nsteps = df['NSTEP'].values

        ########################## Percentage Differences ##########################
        cols_without = [f"ChronoNoneGraphTotalTimeWithout{i}" for i in range(1, num_runs+1)]
        cols_with    = [f"ChronoGraphTotalTimeWithout{i}"     for i in range(1, num_runs+1)]
        
        data_without = df[cols_without].mean(axis=1).values
        data_with = df[cols_with].mean(axis=1).values
        
        # data_without = df[['ChronoNoneGraphTotalTimeWithout1', 'ChronoNoneGraphTotalTimeWithout2', 
        #                                     'ChronoNoneGraphTotalTimeWithout3', 'ChronoNoneGraphTotalTimeWithout4']].mean(axis=1).values
        # data_with = df[['ChronoGraphTotalTimeWithout1', 'ChronoGraphTotalTimeWithout2', 
        #                                  'ChronoGraphTotalTimeWithout3', 'ChronoGraphTotalTimeWithout4']].mean(axis=1).values

        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot Percentage Difference Without First Run/Graph Creation
        plt.plot(
            nsteps, 
            data_without, 
            marker='o', 
            color='red', 
            label='None Graph', 
            linestyle='--'
        )

        # Plot Percentage Difference With First Run/Graph Creation
        plt.plot(
            nsteps, 
            data_with, 
            marker='o', 
            color='red',  # Changed color for distinction
            label='Graph', 
            linestyle='-'
        )

        # Annotate Percentage Difference Without
        for x, y in zip(nsteps, data_without):
            plt.text(
                x, y, f'{y:.2f}ms', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='top'
            )

        # Annotate Percentage Difference With
        for x, y in zip(nsteps, data_with):
            plt.text(
                x, y, f'{y:.2f}ms', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='bottom'
            )
            
        title_test = transform_string(os.path.splitext(os.path.basename(csv_path))[0],"_"," ")

        # plt.title(f'Total Time Difference Percentage in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.title(f'Total Time in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.xlabel('NSTEP (Number of Iterations)')
        plt.ylabel('Total Time (ms)')
        plt.xscale('log')  # Use log scale for NSTEP
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.show()

        # Prepare output filename
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_filename = f"{base_name}_cputotaltime.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save the figure as a JPEG file
        plt.savefig(output_path, format='jpg', dpi=300)
        print(f"Plot saved to {output_path}")

        # Close the figure to free memory
        plt.close()

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")

def generate_launchtotaltime_plot(csv_path, output_dir, num_runs):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure NSTEP is sorted
        df = df.sort_values('NSTEP')

        # Extract NSTEP values
        nsteps = df['NSTEP'].values

        ########################## Percentage Differences ##########################
        # You can choose to plot either 'DiffPercentWithout' or 'DiffPercentWith'
        cols_without = [f"ChronoNoneGraphTotalLaunchTimeWithout{i}" for i in range(1, num_runs+1)]
        cols_with = [f"ChronoGraphTotalLaunchTimeWithout{i}" for i in range(1, num_runs+1)]
        
        data_without = df[cols_without].mean(axis=1).values
        data_with = df[cols_with].mean(axis=1).values

        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot Percentage Difference Without First Run/Graph Creation
        plt.plot(
            nsteps, 
            data_without, 
            marker='o', 
            color='red', 
            label='None Graph', 
            linestyle='--'
        )

        # Plot Percentage Difference With First Run/Graph Creation
        plt.plot(
            nsteps, 
            data_with, 
            marker='o', 
            color='red',  # Changed color for distinction
            label='Graph', 
            linestyle='-'
        )

        # Annotate Percentage Difference Without
        for x, y in zip(nsteps, data_without):
            plt.text(
                x, y, f'{y:.3f}ms', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='top'
            )

        # Annotate Percentage Difference With
        for x, y in zip(nsteps, data_with):
            plt.text(
                x, y, f'{y:.3f}ms', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='bottom'
            )
            
        title_test = transform_string(os.path.splitext(os.path.basename(csv_path))[0],"_"," ")

        # plt.title(f'Time Difference Per Iteration in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.title(f'Total Launch Time in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.xlabel('NSTEP (Number of Iterations)')
        plt.ylabel('Total Launch Time (ms)')
        plt.xscale('log')  # Use log scale for NSTEP
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.show(generate_chronodiffpercent_plot)

        # Prepare output filename
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_filename = f"{base_name}_cputotallaunch.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save the figure as a JPEG file
        plt.savefig(output_path, format='jpg', dpi=300)
        print(f"Plot saved to {output_path}")

        # Close the figure to free memory
        plt.close()

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")

def generate_launchdifftotal_plot(csv_path, output_dir, num_runs):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure NSTEP is sorted
        df = df.sort_values('NSTEP')

        # Extract NSTEP values
        nsteps = df['NSTEP'].values

        ########################## Percentage Differences ##########################
        # You can choose to plot either 'DiffPercentWithout' or 'DiffPercentWith'
        cols_without = [f"ChronoDiffLaunchTimeWithout{i}" for i in range(1, num_runs+1)]
        cols_with = [f"ChronoDiffLaunchTimeWith{i}" for i in range(1, num_runs+1)]
        
        data_without = df[cols_without].mean(axis=1).values
        data_with = df[cols_with].mean(axis=1).values

        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot Percentage Difference Without First Run/Graph Creation
        plt.plot(
            nsteps, 
            data_without, 
            marker='o', 
            color='red', 
            label='Without First Run/Graph', 
            linestyle='-'
        )

        # Plot Percentage Difference With First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     data_with, 
        #     marker='o', 
        #     color='red',  # Changed color for distinction
        #     label='With First Run/Graph', 
        #     linestyle='-'
        # )

        # Annotate Percentage Difference Without
        for x, y in zip(nsteps, data_without):
            plt.text(
                x, y, f'{y:.3f}ms', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='top'
            )

        # Annotate Percentage Difference With
        # for x, y in zip(nsteps, data_with):
        #     plt.text(
        #         x, y, f'{y:.3f}ms', 
        #         fontsize=9, 
        #         color='black', 
        #         ha='left', 
        #         va='bottom'
        #     )
            
        title_test = transform_string(os.path.splitext(os.path.basename(csv_path))[0],"_"," ")

        plt.title(f'CPU Total Launch Time Difference in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.xlabel('NSTEP (Number of Iterations)')
        plt.ylabel('CPU Total Launch Time Difference (ms)')
        plt.xscale('log')  # Use log scale for NSTEP
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.show(generate_chronodiffpercent_plot)

        # Prepare output filename
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_filename = f"{base_name}_cpulaunchdifftotal.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save the figure as a JPEG file
        plt.savefig(output_path, format='jpg', dpi=300)
        print(f"Plot saved to {output_path}")

        # Close the figure to free memory
        plt.close()

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")

# PLOTS NEEDED
# GPU Timer
def generate_gputimeperstep_plot(csv_path, output_dir, num_runs):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure NSTEP is sorted
        df = df.sort_values('NSTEP')

        # Extract NSTEP values
        nsteps = df['NSTEP'].values

        ########################## Percentage Differences ##########################
        # Average total times across runs for 'Without Graph'
        # data_cols_without = ['noneGraphTotalTimeWithout1', 'noneGraphTotalTimeWithout2', 
        #                          'noneGraphTotalTimeWithout3', 'noneGraphTotalTimeWithout4']
        data_cols_without = [f"noneGraphTotalTimeWithout{i}" for i in range(1, num_runs+1)]
        
        mean_without = df[data_cols_without].mean(axis=1).values
        std_without = df[data_cols_without].std(axis=1).values
        time_perstep_without = mean_without / nsteps
        time_perstep_std_without = std_without / nsteps

        # Average total times across runs for 'With Graph'
        # data_cols_with = ['GraphTotalTimeWithout1', 'GraphTotalTimeWithout2', 
        #                 'GraphTotalTimeWithout3', 'GraphTotalTimeWithout4']
        data_cols_with = [f"GraphTotalTimeWithout{i}" for i in range(1, num_runs+1)]
        
        mean_with = df[data_cols_with].mean(axis=1).values
        std_with = df[data_cols_with].std(axis=1).values
        time_perstep_with = (mean_with / nsteps)
        time_perstep_std_with = (std_with / nsteps)
                    
        # Calculate the per step for each nstep

        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot Percentage Difference Without First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     time_perstep_without, 
        #     marker='o', 
        #     color='red', 
        #     label='Without Graph', 
        #     linestyle='--'
        # )
        
        line_obj1 = plt.errorbar(
            nsteps,
            time_perstep_without,
            yerr=time_perstep_std_without,
            marker='o',
            linestyle='--',
            capsize=3,
            color='red',
            label=f"Without Graph"
        )


        # Plot Percentage Difference With First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     time_perstep_with, 
        #     marker='o', 
        #     color='red',  # Changed color for distinction
        #     label='With Graph', 
        #     linestyle='-'
        # )
        
        line_obj2 = plt.errorbar(
            nsteps,
            time_perstep_with,
            yerr=time_perstep_std_with,
            marker='o',
            linestyle='-',
            capsize=3,
            color='red',
            label=f"With Graph"
        )

        # Annotate Percentage Difference Without
        for x, y in zip(nsteps, time_perstep_without):
            plt.text(
                x, y, f'{y:.3f}ms', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='top'
            )

        # Annotate Percentage Difference With
        for x, y in zip(nsteps, time_perstep_with):
            plt.text(
                x, y, f'{y:.3f}ms', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='bottom'
            )
            
        title_test = transform_string(os.path.splitext(os.path.basename(csv_path))[0],"_"," ")

        plt.title(f'GPU Time Per Iteration in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.xlabel('NSTEP (Number of Iterations)')
        plt.ylabel('Time Per Iteration (ms)')
        plt.xscale('log')  # Use log scale for NSTEP
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.show(generate_chronodiffpercent_plot)

        # Prepare output filename
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_filename = f"{base_name}_gputimeperstep.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save the figure as a JPEG file
        plt.savefig(output_path, format='jpg', dpi=300)
        print(f"Plot saved to {output_path}")

        # Close the figure to free memory
        plt.close()

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")

def generate_gpudiffperstep_plot(csv_path, output_dir, num_runs):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure NSTEP is sorted
        df = df.sort_values('NSTEP')

        # Extract NSTEP values
        nsteps = df['NSTEP'].values

        ########################## Percentage Differences ##########################
        # You can choose to plot either 'DiffPercentWithout' or 'DiffPercentWith'
        # data_cols_without = ['DiffPerStepWithout1','DiffPerStepWithout2',
        #                    'DiffPerStepWithout3','DiffPerStepWithout4']
        data_cols_without = [f"DiffPerStepWithout{i}" for i in range(1, num_runs+1)]
        # data_with = df[['DiffPerStepWith1','DiffPerStepWith2',
        #                 'DiffPerStepWith3','DiffPerStepWith4']].mean(axis=1).values
        
        mean_without = df[data_cols_without].mean(axis=1).values
        std_without = df[data_cols_without].std(axis=1).values

        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot Percentage Difference Without First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     mean_without, 
        #     marker='o', 
        #     color='red', 
        #     label='With vs Without Graph', 
        #     linestyle='-'
        # )
        line_obj = plt.errorbar(
            nsteps,
            mean_without,
            yerr=std_without,
            marker='o',
            linestyle='-',
            capsize=3,
            color='red',
            label=f"Δ(With vs Without)"
        )

        # Plot Percentage Difference With First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     data_with, 
        #     marker='o', 
        #     color='red',  # Changed color for distinction
        #     label='With First Run/Graph', 
        #     linestyle='-'
        # )

        # Annotate Percentage Difference Without
        for x, y in zip(nsteps, mean_without):
            plt.text(
                x, y, f'{y:.3f}ms', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='top'
            )

        # Annotate Percentage Difference With
        # for x, y in zip(nsteps, data_with):
        #     plt.text(
        #         x, y, f'{y:.3f}ms', 
        #         fontsize=9, 
        #         color='black', 
        #         ha='left', 
        #         va='bottom'
        #     )
            
        title_test = transform_string(os.path.splitext(os.path.basename(csv_path))[0],"_"," ")

        plt.title(f'GPU Time Difference Per Iteration in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.xlabel('NSTEP (Number of Iterations)')
        plt.ylabel('Time Difference Per Iteration (ms)')
        plt.xscale('log')  # Use log scale for NSTEP
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.show(generate_chronodiffpercent_plot)

        # Prepare output filename
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_filename = f"{base_name}_gpudiffperstep.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save the figure as a JPEG file
        plt.savefig(output_path, format='jpg', dpi=300)
        print(f"Plot saved to {output_path}")

        # Close the figure to free memory
        plt.close()

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")

def generate_gpudiffpercent_plot(csv_path, output_dir, num_runs):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure NSTEP is sorted
        df = df.sort_values('NSTEP')

        # Extract NSTEP values
        nsteps = df['NSTEP'].values

        ########################## Percentage Differences ##########################
        # You can choose to plot either 'DiffPercentWithout' or 'DiffPercentWith'
        # data_cols_without = ['DiffPercentWithout1', 'DiffPercentWithout2', 
        #                                     'DiffPercentWithout3', 'DiffPercentWithout4']
        data_cols_without = [f"DiffPercentWithout{i}" for i in range(1, num_runs+1)]
        # data_with = df[['DiffPercentWith1', 'DiffPercentWith2', 
        #                                  'DiffPercentWith3', 'DiffPercentWith4']].mean(axis=1).values

        mean_without = df[data_cols_without].mean(axis=1).values
        std_without = df[data_cols_without].std(axis=1).values
        
        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot Percentage Difference Without First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     mean_without, 
        #     marker='o', 
        #     color='red', 
        #     label='With vs Without Graph', 
        #     linestyle='-'
        # )
        
        line_obj = plt.errorbar(
            nsteps,
            mean_without,
            yerr=std_without,
            marker='o',
            linestyle='-',
            capsize=3,
            color='red',
            label=f"%Diff(With vs Without)"
        )

        # Plot Percentage Difference With First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     data_with, 
        #     marker='o', 
        #     color='red',  # Changed color for distinction
        #     label='With First Run/Graph', 
        #     linestyle='-'
        # )

        # Annotate Percentage Difference Without
        for x, y in zip(nsteps, mean_without):
            plt.text(
                x, y, f'{y:.2f}%', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='top'
            )

        # Annotate Percentage Difference With
        # for x, y in zip(nsteps, data_with):
        #     plt.text(
        #         x, y, f'{y:.2f}%', 
        #         fontsize=9, 
        #         color='black', 
        #         ha='left', 
        #         va='bottom'
        #     )
            
        title_test = transform_string(os.path.splitext(os.path.basename(csv_path))[0],"_"," ")

        plt.title(f'GPU Total Time Difference Percentage in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.xlabel('NSTEP (Number of Iterations)')
        plt.ylabel('Total Time Difference Percentage (%)')
        plt.xscale('log')  # Use log scale for NSTEP
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.show()

        # Prepare output filename
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_filename = f"{base_name}_gpudiffpercent.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save the figure as a JPEG file
        plt.savefig(output_path, format='jpg', dpi=300)
        print(f"Plot saved to {output_path}")

        # Close the figure to free memory
        plt.close()

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")

#CPU Timer
def generate_cputimeperstep_plot(csv_path, output_dir, num_runs):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure NSTEP is sorted
        df = df.sort_values('NSTEP')

        # Extract NSTEP values
        nsteps = df['NSTEP'].values

        ########################## Percentage Differences ##########################
        # data_cols_without = ['ChronoNoneGraphTotalTimeWithout1', 'ChronoNoneGraphTotalTimeWithout2', 
        #                                     'ChronoNoneGraphTotalTimeWithout3', 'ChronoNoneGraphTotalTimeWithout4']
        data_cols_without = [f"ChronoNoneGraphTotalTimeWithout{i}" for i in range(1, num_runs+1)]
        # time_perstep_without = data_without / nsteps
        mean_without = df[data_cols_without].mean(axis=1).values
        std_without = df[data_cols_without].std(axis=1).values
        time_perstep_without = mean_without / nsteps
        time_perstep_std_without = std_without / nsteps
        
        
        # data_cols_with = ['ChronoGraphTotalTimeWithout1', 'ChronoGraphTotalTimeWithout2', 
        #                                  'ChronoGraphTotalTimeWithout3', 'ChronoGraphTotalTimeWithout4']
        data_cols_with = [f"ChronoGraphTotalTimeWithout{i}" for i in range(1, num_runs+1)]
        # time_perstep_with = data_with / nsteps
        mean_with = df[data_cols_with].mean(axis=1).values
        std_with = df[data_cols_with].std(axis=1).values
        time_perstep_with = mean_with / nsteps
        time_perstep_std_with = std_with / nsteps

        # Calculate the per step for each nstep
        
        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot Percentage Difference Without First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     time_perstep_without, 
        #     marker='o', 
        #     color='red', 
        #     label='Without Graph', 
        #     linestyle='--'
        # )
        
        line_obj1 = plt.errorbar(
            nsteps,
            time_perstep_without,
            yerr=time_perstep_std_without,
            marker='o',
            linestyle='--',
            capsize=3,
            color='red',
            label=f"Without Graph"
        )

        # Plot Percentage Difference With First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     time_perstep_with, 
        #     marker='o', 
        #     color='red',  # Changed color for distinction
        #     label='With Graph', 
        #     linestyle='-'
        # )
        line_obj2 = plt.errorbar(
            nsteps,
            time_perstep_with,
            yerr=time_perstep_std_with,
            marker='o',
            linestyle='-',
            capsize=3,
            color='red',
            label=f"With Graph"
        )

        # Annotate Percentage Difference Without
        for x, y in zip(nsteps, time_perstep_without):
            plt.text(
                x, y, f'{y:.2f}ms', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='top'
            )

        # Annotate Percentage Difference With
        for x, y in zip(nsteps, time_perstep_with):
            plt.text(
                x, y, f'{y:.2f}ms', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='bottom'
            )
            
        title_test = transform_string(os.path.splitext(os.path.basename(csv_path))[0],"_"," ")

        plt.title(f'CPU Time Per Iteration in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.xlabel('NSTEP (Number of Iterations)')
        plt.ylabel('Time Per Iteration (ms)')
        plt.xscale('log')  # Use log scale for NSTEP
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.show()

        # Prepare output filename
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        # base_name = ("_").join(base_name.split("_")[:-1])
        output_filename = f"{base_name}_cputimeperstep.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save the figure as a JPEG file
        plt.savefig(output_path, format='jpg', dpi=300)
        print(f"Plot saved to {output_path}")

        # Close the figure to free memory
        plt.close()

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")

def generate_cpudiffperstep_plot(csv_path, output_dir, num_runs):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure NSTEP is sorted
        df = df.sort_values('NSTEP')

        # Extract NSTEP values
        nsteps = df['NSTEP'].values

        ########################## Percentage Differences ##########################
        # data_cols_without = ['ChronoDiffPerStepWithout1','ChronoDiffPerStepWithout2',
        #                    'ChronoDiffPerStepWithout3','ChronoDiffPerStepWithout4']
        data_cols_without = [f"ChronoDiffPerStepWithout{i}" for i in range(1, num_runs+1)]
        # data_with = df[['ChronoDiffPerStepWith1','ChronoDiffPerStepWith2',
        #                 'ChronoDiffPerStepWith3','ChronoDiffPerStepWith4']].mean(axis=1).values

        mean_without = df[data_cols_without].mean(axis=1).values
        std_without = df[data_cols_without].std(axis=1).values
        
        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot Percentage Difference Without First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     data_without, 
        #     marker='o', 
        #     color='red', 
        #     label='With vs Without Graph', 
        #     linestyle='-'
        # )
        line_obj = plt.errorbar(
            nsteps,
            mean_without,
            yerr=std_without,
            marker='o',
            linestyle='-',
            capsize=3,
            color='red',
            label=f"Δ(With vs Without)"
        )

        # Plot Percentage Difference With First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     data_with, 
        #     marker='o', 
        #     color='red',  # Changed color for distinction
        #     label='With First Run/Graph', 
        #     linestyle='-'
        # )

        # Annotate Percentage Difference Without
        for x, y in zip(nsteps, mean_without):
            plt.text(
                x, y, f'{y:.3f}ms', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='top'
            )

        # Annotate Percentage Difference With
        # for x, y in zip(nsteps, data_with):
        #     plt.text(
        #         x, y, f'{y:.3f}ms', 
        #         fontsize=9, 
        #         color='black', 
        #         ha='left', 
        #         va='bottom'
        #     )
            
        title_test = transform_string(os.path.splitext(os.path.basename(csv_path))[0],"_"," ")

        plt.title(f'CPU Time Difference Per Iteration in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.xlabel('NSTEP (Number of Iterations)')
        plt.ylabel('Time Difference Per Iteration (ms)')
        plt.xscale('log')  # Use log scale for NSTEP
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.show(generate_chronodiffpercent_plot)

        # Prepare output filename
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_filename = f"{base_name}_cpudiffperstep.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save the figure as a JPEG file
        plt.savefig(output_path, format='jpg', dpi=300)
        print(f"Plot saved to {output_path}")

        # Close the figure to free memory
        plt.close()

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")

def generate_cpudiffpercent_plot(csv_path, output_dir, num_runs):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure NSTEP is sorted
        df = df.sort_values('NSTEP')

        # Extract NSTEP values
        nsteps = df['NSTEP'].values

        ########################## Percentage Differences ##########################
        # data_cols_without = ['ChronoDiffPercentWithout1','ChronoDiffPercentWithout2',
        #                    'ChronoDiffPercentWithout3','ChronoDiffPercentWithout4']
        data_cols_without = [f"ChronoDiffPercentWithout{i}" for i in range(1, num_runs+1)]
        # data_with = df[['ChronoDiffPercentWith1','ChronoDiffPercentWith2',
        #                 'ChronoDiffPercentWith3','ChronoDiffPercentWith4']].mean(axis=1).values
        mean_without = df[data_cols_without].mean(axis=1).values
        std_without = df[data_cols_without].std(axis=1).values
        
        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot Percentage Difference Without First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     data_without, 
        #     marker='o', 
        #     color='red', 
        #     label='With vs Without Graph', 
        #     linestyle='-'
        # )
        line_obj = plt.errorbar(
            nsteps,
            mean_without,
            yerr=std_without,
            marker='o',
            linestyle='-',
            capsize=3,
            color='red',
            label=f"%Diff(With vs Without)"
        )

        # Plot Percentage Difference With First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     data_with, 
        #     marker='o', 
        #     color='red',  # Changed color for distinction
        #     label='With First Run/Graph', 
        #     linestyle='-'
        # )

        # Annotate Percentage Difference Without
        for x, y in zip(nsteps, mean_without):
            plt.text(
                x, y, f'{y:.2f}%', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='top'
            )

        # Annotate Percentage Difference With
        # for x, y in zip(nsteps, data_with):
        #     plt.text(
        #         x, y, f'{y:.2f}%', 
        #         fontsize=9, 
        #         color='black', 
        #         ha='left', 
        #         va='bottom'
        #     )
            
        title_test = transform_string(os.path.splitext(os.path.basename(csv_path))[0],"_"," ")

        plt.title(f'CPU Total Time Difference Percentage in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.xlabel('NSTEP (Number of Iterations)')
        plt.ylabel('Total Time Difference Percentage (%)')
        plt.xscale('log')  # Use log scale for NSTEP
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.show()

        # Prepare output filename
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        # base_name = ("_").join(base_name.split("_")[:-1])
        output_filename = f"{base_name}_cpudiffpercent.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save the figure as a JPEG file
        plt.savefig(output_path, format='jpg', dpi=300)
        print(f"Plot saved to {output_path}")

        # Close the figure to free memory
        plt.close()

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")


#LAUNCH Timer
def generate_launchtimeperstep_plot(csv_path, output_dir, num_runs):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure NSTEP is sorted
        df = df.sort_values('NSTEP')

        # Extract NSTEP values
        nsteps = df['NSTEP'].values

        ########################## Percentage Differences ##########################
        
        # data_cols_without = ['ChronoNoneGraphTotalLaunchTimeWithout1','ChronoNoneGraphTotalLaunchTimeWithout2',
        #                   'ChronoNoneGraphTotalLaunchTimeWithout3','ChronoNoneGraphTotalLaunchTimeWithout4']
        data_cols_without = [f"ChronoNoneGraphTotalLaunchTimeWithout{i}" for i in range(1, num_runs+1)]
        # time_perstep_without = data_without / nsteps
        mean_without = df[data_cols_without].mean(axis=1).values
        std_without = df[data_cols_without].std(axis=1).values
        time_perstep_without = (mean_without / nsteps) * 1000
        time_perstep_std_without = (std_without / nsteps) * 1000
        
        # data_cols_with = ['ChronoGraphTotalLaunchTimeWithout1','ChronoGraphTotalLaunchTimeWithout2',
        #                 'ChronoGraphTotalLaunchTimeWithout3','ChronoGraphTotalLaunchTimeWithout4']
        data_cols_with = [f"ChronoGraphTotalLaunchTimeWithout{i}" for i in range(1, num_runs+1)]
        # time_perstep_with = data_with / nsteps
        mean_with = df[data_cols_with].mean(axis=1).values
        std_with = df[data_cols_with].std(axis=1).values
        time_perstep_with = (mean_with / nsteps) * 1000
        time_perstep_std_with = (std_with / nsteps) * 1000
        
        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot Percentage Difference Without First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     time_perstep_without, 
        #     marker='o', 
        #     color='red', 
        #     label='Without Graph', 
        #     linestyle='--'
        # )
        line_obj1 = plt.errorbar(
            nsteps,
            time_perstep_without,
            yerr=time_perstep_std_without,
            marker='o',
            linestyle='--',
            capsize=3,
            color='red',
            label=f"Without Graph"
        )

        # Plot Percentage Difference With First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     time_perstep_with, 
        #     marker='o', 
        #     color='red',  # Changed color for distinction
        #     label='With Graph', 
        #     linestyle='-'
        # )
        line_obj2 = plt.errorbar(
            nsteps,
            time_perstep_with,
            yerr=time_perstep_std_with,
            marker='o',
            linestyle='-',
            capsize=3,
            color='red',
            label=f"With Graph"
        )

        # Annotate Percentage Difference Without
        for x, y in zip(nsteps, time_perstep_without):
            plt.text(
                x, y, f'{y:.2f}', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='top'
            )

        # Annotate Percentage Difference With
        for x, y in zip(nsteps, time_perstep_with):
            plt.text(
                x, y, f'{y:.2f}', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='bottom'
            )
            
        title_test = transform_string(os.path.splitext(os.path.basename(csv_path))[0],"_"," ")

        plt.title(f'Launch Time Per Iteration in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.xlabel('NSTEP (Number of Iterations)')
        # plt.ylabel('Time Per Iteration (ms)')
        plt.ylabel('Time Per Iteration (μs)')
        plt.xscale('log')  # Use log scale for NSTEP
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.show(generate_chronodiffpercent_plot)

        # Prepare output filename
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_filename = f"{base_name}_launchtimeperstep.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save the figure as a JPEG file
        plt.savefig(output_path, format='jpg', dpi=300)
        print(f"Plot saved to {output_path}")

        # Close the figure to free memory
        plt.close()

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")

def generate_launchdiffperstep_plot(csv_path, output_dir, num_runs):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure NSTEP is sorted
        df = df.sort_values('NSTEP')

        # Extract NSTEP values
        nsteps = df['NSTEP'].values

        ########################## Percentage Differences ##########################
        # data_cols_without = ['ChronoDiffLaunchTimeWithout1','ChronoDiffLaunchTimeWithout2',
        #                   'ChronoDiffLaunchTimeWithout3','ChronoDiffLaunchTimeWithout4']
        data_cols_without = [f"ChronoDiffLaunchTimeWithout{i}" for i in range(1, num_runs+1)]
        # time_perstep_without = data_without / nsteps
        mean_without = df[data_cols_without].mean(axis=1).values
        std_without = df[data_cols_without].std(axis=1).values
        time_perstep_without = (mean_without / nsteps) * 1000
        time_perstep_std_without = (std_without / nsteps) * 1000
        
        
        # data_with = df[['ChronoDiffLaunchTimeWith1','ChronoDiffLaunchTimeWith2',
        #                 'ChronoDiffLaunchTimeWith3','ChronoDiffLaunchTimeWith4']].mean(axis=1).values
        # time_perstep_with = data_with / nsteps

        # Plotting
        plt.figure(figsize=(plot_w, plot_h))

        # Plot Percentage Difference Without First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     time_perstep_without, 
        #     marker='o', 
        #     color='red', 
        #     label='Without Graph', 
        #     linestyle='-'
        # )
        line_obj = plt.errorbar(
            nsteps,
            time_perstep_without,
            yerr=time_perstep_std_without,
            marker='o',
            linestyle='-',
            capsize=3,
            color='red',
            label=f"Δ(With vs Without)"
        )

        # Plot Percentage Difference With First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     time_perstep_with, 
        #     marker='o', 
        #     color='red',  # Changed color for distinction
        #     label='With First Run/Graph', 
        #     linestyle='-'
        # )

        # Annotate Percentage Difference Without
        for x, y in zip(nsteps, time_perstep_without):
            plt.text(
                x, y, f'{y:.1f}', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='top'
            )

        # Annotate Percentage Difference With
        # for x, y in zip(nsteps, time_perstep_with):
        #     plt.text(
        #         x, y, f'{y:.3f}ms', 
        #         fontsize=9, 
        #         color='black', 
        #         ha='left', 
        #         va='bottom'
        #     )
            
        title_test = transform_string(os.path.splitext(os.path.basename(csv_path))[0],"_"," ")

        plt.title(f'Launch Total Time Difference in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.xlabel('NSTEP (Number of Iterations)')
        plt.ylabel('Total Time Difference (μs)')
        plt.xscale('log')  # Use log scale for NSTEP
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.show(generate_chronodiffpercent_plot)

        # Prepare output filename
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        output_filename = f"{base_name}_launchdiffperstep.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save the figure as a JPEG file
        plt.savefig(output_path, format='jpg', dpi=300)
        print(f"Plot saved to {output_path}")

        # Close the figure to free memory
        plt.close()

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")

def generate_launchdiffpercent_plot(csv_path, output_dir, num_runs):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)

        # Ensure NSTEP is sorted
        df = df.sort_values('NSTEP')

        # Extract NSTEP values
        nsteps = df['NSTEP'].values

        ########################## Percentage Differences ##########################
        # You can choose to plot either 'DiffPercentWithout' or 'DiffPercentWith'
        # data_cols_without = ['ChronoDiffLaunchPercentWithout1','ChronoDiffLaunchPercentWithout2',
        #                    'ChronoDiffLaunchPercentWithout3','ChronoDiffLaunchPercentWithout4']
        data_cols_without = [f"ChronoDiffLaunchPercentWithout{i}" for i in range(1, num_runs+1)]
        # data_with = df[['ChronoDiffLaunchPercentWith1','ChronoDiffLaunchPercentWith2',
        #                 'ChronoDiffLaunchPercentWith3','ChronoDiffLaunchPercentWith4']].mean(axis=1).values
        mean_without = df[data_cols_without].mean(axis=1).values
        std_without = df[data_cols_without].std(axis=1).values
        
        
        
        # Plotting
        plt.figure(figsize=(8, 6))

        # Plot Percentage Difference Without First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     data_without, 
        #     marker='o', 
        #     color='red', 
        #     label='With vs Without Graph', 
        #     linestyle='-'
        # )
        line_obj = plt.errorbar(
            nsteps,
            mean_without,
            yerr=std_without,
            marker='o',
            linestyle='-',
            capsize=3,
            color='red',
            label=f"%Diff(With vs Without)"
        )

        # Plot Percentage Difference With First Run/Graph Creation
        # plt.plot(
        #     nsteps, 
        #     data_with, 
        #     marker='o', 
        #     color='red',  # Changed color for distinction
        #     label='With First Run/Graph', 
        #     linestyle='-'
        # )

        # Annotate Percentage Difference Without
        for x, y in zip(nsteps, mean_without):
            plt.text(
                x, y, f'{y:.2f}%', 
                fontsize=9, 
                color='black', 
                ha='left', 
                va='top'
            )

        # Annotate Percentage Difference With
        # for x, y in zip(nsteps, data_with):
        #     plt.text(
        #         x, y, f'{y:.2f}%', 
        #         fontsize=9, 
        #         color='black', 
        #         ha='left', 
        #         va='bottom'
        #     )
            
        title_test = transform_string(os.path.splitext(os.path.basename(csv_path))[0],"_"," ")

        plt.title(f'Launch Total Time Difference Percentage in CUDA (NVIDIA Tesla T4): {title_test} Test', fontsize=font_size)
        plt.xlabel('NSTEP (Number of Iterations)')
        plt.ylabel('Total Time Difference Percentage (%)')
        plt.xscale('log')  # Use log scale for NSTEP
        plt.grid(True, which='both', linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        # plt.show()

        # Prepare output filename
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        # base_name = ("_").join(base_name.split("_")[:-1])
        output_filename = f"{base_name}_launchdiffpercent.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save the figure as a JPEG file
        plt.savefig(output_path, format='jpg', dpi=300)
        print(f"Plot saved to {output_path}")

        # Close the figure to free memory
        plt.close()

    except Exception as e:
        print(f"Failed to process {csv_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate performance improvement plots from CSV files.'
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
        default='output_plots',
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

    # Process each CSV file
    for csv_path in args.csv_files:
        if not os.path.isfile(csv_path):
            print(f"File not found: {csv_path}")
            continue
        #GPU
        generate_gputimeperstep_plot(csv_path, output_dir, args.num_runs)
        generate_gpudiffperstep_plot(csv_path, output_dir, args.num_runs)
        generate_gpudiffpercent_plot(csv_path, output_dir, args.num_runs)
        #CPU
        generate_cputimeperstep_plot(csv_path, output_dir, args.num_runs)
        generate_cpudiffperstep_plot(csv_path, output_dir, args.num_runs)
        generate_cpudiffpercent_plot(csv_path, output_dir, args.num_runs)
        #LAUNCH
        generate_launchtimeperstep_plot(csv_path, output_dir, args.num_runs)
        generate_launchdiffperstep_plot(csv_path, output_dir, args.num_runs)
        generate_launchdiffpercent_plot(csv_path, output_dir, args.num_runs)
        
        #EXTRAS   
        # generate_cputotaltime_plot(csv_path, output_dir)
        # generate_launchtotaltime_plot(csv_path, output_dir)
        # generate_launchdifftotal_plot(csv_path, output_dir)
        

if __name__ == "__main__":
    main()

