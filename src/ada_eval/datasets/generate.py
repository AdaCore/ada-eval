import shutil
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from ada_eval.datasets.loader import load_packed_dataset
from ada_eval.datasets.types.datasets import Dataset, get_packed_dataset_files
from ada_eval.datasets.types.samples import GeneratedSample, Sample
from ada_eval.paths import GENERATION_WORKING_DIR
from ada_eval.tools.generic_tool import GenericTool


def get_dataset_working_dir(dataset: Dataset) -> Path:
    return GENERATION_WORKING_DIR / f"{dataset.type}_{dataset.name}"


def get_sample_working_dir(sample: Sample, dataset_working_dir: Path) -> Path:
    return dataset_working_dir / sample.name


def unpack_dataset_for_generation(dataset: Dataset):
    # Remove files from previous runs
    dataset_working_dir = get_dataset_working_dir(dataset)
    shutil.rmtree(dataset_working_dir, ignore_errors=True)

    # Don't ignore errors so we know if it fails to clean up previous files
    dataset_working_dir.mkdir()

    # Unpack each sample
    for sample in dataset.samples:
        sample_working_dir = get_sample_working_dir(sample, dataset_working_dir)
        sample_working_dir.mkdir()
        sample.unpack_for_generation(sample_working_dir)


@dataclass
class InProgressSample:
    sample: Sample
    working_dir: Path


def generate_completions(
    packed_dataset_or_dir: Path, jobs: int, tool: GenericTool, output_dir: Path
):
    dataset_files = get_packed_dataset_files(packed_dataset_or_dir)

    if len(dataset_files) == 0:
        print(f"No datasets could be found at: {packed_dataset_or_dir}")
        return

    GENERATION_WORKING_DIR.mkdir(exist_ok=True)
    datasets = [load_packed_dataset(path) for path in dataset_files]
    datasets = [x for x in datasets if x.type in tool.supported_dataset_types()]
    if len(datasets) == 0:
        print(
            f"No datasets supported by {tool.name} could be found at:",
            packed_dataset_or_dir,
        )
        return

    # Unpack all of the samples
    samples: dict[Dataset, list[InProgressSample]] = {}
    for dataset in datasets:
        unpack_dataset_for_generation(dataset)
        dataset_wd = get_dataset_working_dir(dataset)
        samples[dataset] = [
            InProgressSample(sample, get_sample_working_dir(sample, dataset_wd))
            for sample in dataset.samples
        ]

    # Generate completions for each sample
    dataset_results: dict[Dataset, list[GeneratedSample]] = {}

    # Calculate total number of samples for progress tracking
    total_samples = sum(len(sample_list) for sample_list in samples.values())

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        # Submit all futures and create a mapping from future to dataset
        future_to_dataset: dict[Future[GeneratedSample], Dataset] = {}
        for dataset, in_progress_samples in samples.items():
            dataset_results[dataset] = []
            for sample in in_progress_samples:
                future = executor.submit(tool.apply, sample.working_dir, sample.sample)
                future_to_dataset[future] = dataset

        # Process futures as they complete with progress tracking
        with tqdm(
            total=total_samples,
            desc="Generating completions",
        ) as pbar:
            for future in as_completed(future_to_dataset.keys()):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    dataset_results[dataset].append(result)
                except Exception as e:  # noqa: BLE001 we want to catch any and all exceptions
                    print(f"Error processing sample: {e}")
                finally:
                    pbar.update(1)

    # Write the results to file
    for dataset, results in dataset_results.items():
        output_file = output_dir / f"{dataset.type}_{dataset.name}.jsonl"
        with output_file.open("w") as f:
            for result in results:
                f.write(result.model_dump_json() + "\n")
