# from subprocess import Popen, PIPE, TimeoutExpired
# from dataclasses import dataclass
# import psutil
# from ada_eval.datasets.types.samples import SampleResult


# def run(command, cwd, timeout_s=None) -> SampleResult:
#     runner_timeout = False

#     with Popen(command, stdout=PIPE, stderr=PIPE, encoding="utf-8", cwd=cwd) as process:
#         p = psutil.Process(process.pid)
#         try:
#             stdout, stderr = process.communicate(input, timeout=timeout_s)
#         except TimeoutExpired:
#             runner_timeout = True
#             process.kill()
#             stdout, stderr = process.communicate()
#         except:
#             process.kill()
#             raise
#         retcode = process.poll()
#         with p.oneshot():
#             cpu_times = p.cpu_times()

#     print(cpu_times)

#     return Result(
#         stdout=stdout,
#         stderr=stderr,
#         retcode=retcode,
#         runner_timeout=runner_timeout
#     )
