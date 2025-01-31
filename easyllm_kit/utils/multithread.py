import concurrent.futures
from typing import Callable, List, Iterator, Any, Optional
import tqdm

def run_tasks_in_multithreads(
        func: Callable[[Any], Any],
        tasks: List[Any],
        num_workers: int = 5,
        use_progress_bar: bool = False,
        progress_bar_desc: Optional[str] = None
) -> Iterator[Any]:
    """
    Executes a function in parallel using multiple threads and yields results as they complete.

    Args:
        func: The function to run. The function must accept a single argument.
        tasks: A list of tasks i.e. arguments to func.
        num_workers: Maximum number of parallel workers.
        use_progress_bar: Whether to use a progress bar. Default False.
        progress_bar_desc: String to display in the progress bar. Default None.

    Yields:
        The result of executing the function for each task.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {executor.submit(func, task): task for task in tasks}
        
        if use_progress_bar:
            pbar = tqdm.tqdm(total=len(tasks), desc=progress_bar_desc, dynamic_ncols=True)
        
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                yield result
            except Exception as e:
                print(f"Task {task} generated an exception: {e}")
                yield None
            finally:
                if use_progress_bar:
                    pbar.update(1)
        
        if use_progress_bar:
            pbar.close()
