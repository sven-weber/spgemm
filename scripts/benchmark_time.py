
def graph_single_run(folder: str):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='benchmark_time',
                    description='Generate graphs from benchmarks')
    parser.add_argument('--run', type=str, required=False, default='')
    parser.add_argument('--multiple-runs', type=str, nargs='+', default=[])
    args = parser.parse_args()

    if len(args.run) > 0:
        graph_single_run(args.runs)
    elif len(args.multiple_runs) > 0:
        graph_multiple_runs(args.multiple_runs)
    else:
        print("You must specify either --run or --runs")
        exit(1)
