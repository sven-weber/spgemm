import os
import pathlib
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

N_WARMUP = 5
N_RUNS = 50
SHUFFLING = "none"
PARTITIONING = "balanced"
SCRATCH = os.getenv("SCRATCH")
CMD = "/build/dphpc"
CMD = os.path.join(SCRATCH, "spgemm/build/dphpc")
RUNS_DIR = "runs"
RUNS_DIR = os.path.join(SCRATCH, "spgemm/runs")

cpus_per_machine = 64
threads_per_mpi = 16
MPI_OPEN_MP_CONFIG = [
    {"nodes": (16 * threads_per_mpi) // cpus_per_machine, "mpi": 16},
    {"nodes": (64 * threads_per_mpi) // cpus_per_machine, "mpi": 64},
    {"nodes": (256 * threads_per_mpi) // cpus_per_machine, "mpi": 256},
    {"nodes": (512 * threads_per_mpi) // cpus_per_machine, "mpi": 512},
]

def render_batch_script(impls, matrices, mpi, nodes, last_job=None):
    if "comb3d" in impls:
        mpi = mpi*8

    processes_per_mpi = (nodes * cpus_per_machine) // mpi
    n_switches = 1 if nodes <= 256 else 2
    date = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    log_out = f"{RUNS_DIR}/slurm_{date}.out"
    matrix_path = os.path.join(SCRATCH, "matrices")

    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("./scripts/job_template2.sh.j2")
    script = template.render(
        cmd=CMD,
        impls=impls,
        matrices=matrices,
        mpi_processes=mpi,
        n_machines=nodes,
        processes_per_mpi=processes_per_mpi,
        matrix_path=matrix_path,
        n_runs=N_RUNS,
        n_warmup=N_WARMUP,
        shuffling=SHUFFLING,
        partitioning=PARTITIONING,
        parallel_load_str="false",
        persist_str="false",
        runs_dir=RUNS_DIR,
        switches=n_switches,
        log_out=log_out,
        last_job=last_job
    )

    #job_path = f"jobs/job_{nodes}_{mpi}{"_comb3d" if "comb3d" in impls else ""}.sh"
    job_path = f"jobs/job_{nodes}_{mpi}{'_comb3d' if 'comb3d' in impls else ''}.sh"
    pathlib.Path("jobs").mkdir(parents=True, exist_ok=True)
    with open(job_path, "w") as f:
        f.write(script)
    os.chmod(job_path, 0o755)
    print(f"Generated script: {job_path}")

def main():
    matrices = ["dielFilterV2real", "stokes", "HV15R", "nlpkkt200", "nlpkkt240", "GAP-twitter"]
    matrices = '"dielFilterV2real stokes HV15R nlpkkt200 nlpkkt240 GAP-twitter"'

    impls = ["comb1d", "comb2d", "drop_parallel"]
    impls = '"comb1d comb2d drop_parallel"'
    for config in MPI_OPEN_MP_CONFIG:
        render_batch_script(impls, matrices, config["mpi"], config["nodes"])
    
    impls = "comb3d"
    for config in MPI_OPEN_MP_CONFIG:
        render_batch_script(impls, matrices, config["mpi"], config["nodes"])

if __name__ == "__main__":
    main()
