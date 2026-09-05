from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


# ======================== 只需修改这里的实验配置 ========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 数据文件路径：可以使用绝对路径，也可以使用相对于项目根目录的路径。
MNIST_SOURCE = PROJECT_ROOT / "data" / "mnist_train.npy"
CCAT_SOURCE = PROJECT_ROOT / "data" / "ccat_test_csr.npz"

# 决定本次运行哪些实验。首次检查建议只开启人工数据。
RUN_SYNTHETIC = True
RUN_MNIST = False
RUN_CCAT_BENCHMARK = False
RUN_FULL_CCAT = False

# 是否在实验完成后自动生成收敛图。
PLOT_MNIST = True
PLOT_CCAT = True

# 公共随机设置。每次重复使用可记录的独立种子。
BASE_SEED = 20260724
REPEATS = 3

# MNIST 设置。
MNIST_SAMPLES = 60000
MNIST_RANKS = [3, 6, 17, 28]
MNIST_EPOCHS = 30
MNIST_EPOCH_LENGTH = 60000

# CCAT 子集统计实验设置。把 CCAT_SAMPLES 设为 0 表示使用全部样本。
CCAT_SAMPLES = 80000
CCAT_RANKS = [6, 16, 32]
CCAT_EPOCHS = 30
CCAT_EPOCH_LENGTH = 80000

# 完整 CCAT 原论文协议设置；该实验耗时很长，只有 RUN_FULL_CCAT=True 时运行。
FULL_CCAT_RANK = 6
FULL_CCAT_EPOCHS = 30
FULL_CCAT_WARM_PASSES = 0

# 人工数据设置。
SYNTHETIC_SAMPLES = 20000
SYNTHETIC_FEATURES = 100
SYNTHETIC_RANK = 10
SYNTHETIC_EPOCHS = 30
SYNTHETIC_EPOCH_LENGTH = 100
SYNTHETIC_GAPS = [0.008, 0.016, 0.032]

# 为保证计时公平，默认把常见线性代数库限制为单线程。
THREADS = 1
# ======================================================================


@dataclass
class OutputFolders:
    root: Path
    results: Path
    figures: Path
    logs: Path
    metadata: Path


def create_output_folders() -> OutputFolders:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    root = PROJECT_ROOT / "output" / "main_experiments" / run_name
    folders = OutputFolders(
        root=root,
        results=root / "results",
        figures=root / "figures",
        logs=root / "logs",
        metadata=root / "metadata",
    )
    for folder in asdict(folders).values():
        Path(folder).mkdir(parents=True, exist_ok=True)
    return folders


def experiment_environment() -> dict[str, str]:
    environment = os.environ.copy()
    thread_value = str(THREADS)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = thread_value
    environment["MPLCONFIGDIR"] = str(PROJECT_ROOT / "output" / ".matplotlib")
    return environment


def run_command(name: str, arguments: list[str], folders: OutputFolders,
                environment: dict[str, str]) -> None:
    command = [sys.executable, *arguments]
    log_path = folders.logs / f"{name}.log"
    print(f"\n[{name}] 开始运行")
    print("命令：", " ".join(command))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError(f"无法读取 {name} 的运行输出")
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{name} 运行失败，详情见 {log_path}")
    print(f"[{name}] 完成，日志：{log_path}")


def require_data(name: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"需要运行 {name}，但找不到数据文件：{path}\n"
            "请修改 main_experiments.py 顶部对应的 SOURCE 路径。"
        )


def collect_figures(source: Path, dataset: str, folders: OutputFolders) -> None:
    destination = folders.figures / dataset
    destination.mkdir(parents=True, exist_ok=True)
    for figure in source.glob("*.pdf"):
        shutil.copy2(figure, destination / figure.name)


def benchmark_arguments(dataset: str, source: Path, samples: int,
                        ranks: list[int], epochs: int, epoch_length: int,
                        output: Path) -> list[str]:
    return [
        str(SCRIPT_DIR / "run_benchmarks.py"),
        "--dataset", dataset,
        "--source", str(source),
        "--samples", str(samples),
        "--ranks", *[str(rank) for rank in ranks],
        "--epochs", str(epochs),
        "--epoch-length", str(epoch_length),
        "--repeats", str(REPEATS),
        "--seed", str(BASE_SEED),
        "--output", str(output),
    ]


def plot_arguments(dataset: str, source: Path, samples: int,
                   ranks: list[int], output: Path) -> list[str]:
    return [
        str(SCRIPT_DIR / "plot_convergence.py"),
        "--dataset", dataset,
        "--source", str(source),
        "--samples", str(samples),
        "--ranks", *[str(rank) for rank in ranks],
        "--seed", str(BASE_SEED),
        "--output", str(output),
    ]


def save_configuration(folders: OutputFolders) -> None:
    configuration = {
        "project_root": str(PROJECT_ROOT),
        "mnist_source": str(MNIST_SOURCE),
        "ccat_source": str(CCAT_SOURCE),
        "switches": {
            "run_synthetic": RUN_SYNTHETIC,
            "run_mnist": RUN_MNIST,
            "run_ccat_benchmark": RUN_CCAT_BENCHMARK,
            "run_full_ccat": RUN_FULL_CCAT,
        },
        "base_seed": BASE_SEED,
        "repeats": REPEATS,
        "threads": THREADS,
        "mnist": {
            "samples": MNIST_SAMPLES, "ranks": MNIST_RANKS,
            "epochs": MNIST_EPOCHS, "epoch_length": MNIST_EPOCH_LENGTH,
        },
        "ccat": {
            "samples": CCAT_SAMPLES, "ranks": CCAT_RANKS,
            "epochs": CCAT_EPOCHS, "epoch_length": CCAT_EPOCH_LENGTH,
        },
        "synthetic": {
            "samples": SYNTHETIC_SAMPLES, "features": SYNTHETIC_FEATURES,
            "rank": SYNTHETIC_RANK, "epochs": SYNTHETIC_EPOCHS,
            "epoch_length": SYNTHETIC_EPOCH_LENGTH, "gaps": SYNTHETIC_GAPS,
        },
    }
    path = folders.metadata / "experiment_configuration.json"
    path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")


def main() -> None:
    folders = create_output_folders()
    environment = experiment_environment()
    save_configuration(folders)
    run_command(
        "environment",
        [str(SCRIPT_DIR / "environment_report.py"),
         "--output", str(folders.metadata / "environment.json")],
        folders,
        environment,
    )

    if RUN_SYNTHETIC:
        synthetic_output = folders.results / "synthetic"
        run_command(
            "synthetic",
            [
                str(SCRIPT_DIR / "run_synthetic.py"),
                "--output", str(synthetic_output),
                "--samples", str(SYNTHETIC_SAMPLES),
                "--features", str(SYNTHETIC_FEATURES),
                "--rank", str(SYNTHETIC_RANK),
                "--epochs", str(SYNTHETIC_EPOCHS),
                "--epoch-length", str(SYNTHETIC_EPOCH_LENGTH),
                "--repeats", str(REPEATS),
                "--seed", str(BASE_SEED),
                "--gaps", *[str(gap) for gap in SYNTHETIC_GAPS],
            ],
            folders,
            environment,
        )
        collect_figures(synthetic_output, "synthetic", folders)

    if RUN_MNIST:
        require_data("MNIST", MNIST_SOURCE)
        mnist_output = folders.results / "mnist"
        run_command(
            "mnist_benchmark",
            benchmark_arguments("mnist", MNIST_SOURCE, MNIST_SAMPLES,
                                MNIST_RANKS, MNIST_EPOCHS,
                                MNIST_EPOCH_LENGTH, mnist_output),
            folders,
            environment,
        )
        if PLOT_MNIST:
            run_command(
                "mnist_convergence_figures",
                plot_arguments("mnist", MNIST_SOURCE, MNIST_SAMPLES,
                               MNIST_RANKS, mnist_output),
                folders,
                environment,
            )
        collect_figures(mnist_output, "mnist", folders)

    if RUN_CCAT_BENCHMARK:
        require_data("CCAT", CCAT_SOURCE)
        ccat_output = folders.results / "ccat"
        run_command(
            "ccat_benchmark",
            benchmark_arguments("ccat", CCAT_SOURCE, CCAT_SAMPLES,
                                CCAT_RANKS, CCAT_EPOCHS,
                                CCAT_EPOCH_LENGTH, ccat_output),
            folders,
            environment,
        )
        if PLOT_CCAT:
            run_command(
                "ccat_convergence_figures",
                plot_arguments("ccat", CCAT_SOURCE, CCAT_SAMPLES,
                               CCAT_RANKS, ccat_output),
                folders,
                environment,
            )
        collect_figures(ccat_output, "ccat", folders)

    if RUN_FULL_CCAT:
        require_data("完整 CCAT", CCAT_SOURCE)
        run_command(
            "full_ccat",
            [
                str(SCRIPT_DIR / "run_ccat_original_protocol.py"),
                "--data", str(CCAT_SOURCE),
                "--output", str(folders.results / "full_ccat"),
                "--rank", str(FULL_CCAT_RANK),
                "--epochs", str(FULL_CCAT_EPOCHS),
                "--method", "both",
                "--orth-backend", "paper",
                "--warm-passes", str(FULL_CCAT_WARM_PASSES),
                "--independent-samples",
            ],
            folders,
            environment,
        )

    print("\n全部已开启实验运行完成。")
    print("本次结果目录：", folders.root)
    print("数值结果：", folders.results)
    print("图表：", folders.figures)
    print("日志：", folders.logs)
    print("环境和参数：", folders.metadata)


if __name__ == "__main__":
    main()
