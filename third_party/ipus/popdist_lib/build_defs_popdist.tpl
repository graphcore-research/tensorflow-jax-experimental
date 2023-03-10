"""Macros for supporting PopDist with OpenMPI."""

def poprun_py_test(
        name,
        srcs,
        main,
        num_replicas = 1,
        num_instances = 1,
        ipus_per_replica = 1,
        args = [],
        tags = [],
        **kwargs):
    native.py_test(
        name = name,
        srcs = srcs + [
            "//third_party/ipus/popdist_lib:poprun_test_wrapper.py",
        ],
        main = "poprun_test_wrapper.py",
        # All poprun tests use hardware and must be exclusive to avoid other
        # tests from interfering. With other tests running in parallel there
        # is a race condition in which another test could acquire a device
        # between the parent device configuration (by poprun) and the child
        # device acquisition (by the instances).
        tags = tags + ["exclusive", "hw_poplar_test_16_ipus"],
        args = [
            "$(location @local_config_poplar//poplar:mpirun)",
            "$(location @local_config_poplar//poplar:poprun)",
            "--num-replicas",
            str(num_replicas),
            "--num-instances",
            str(num_instances),
            "--ipus-per-replica",
            str(ipus_per_replica),
            "--process-placement=disabled",
            "--mpi-global-args=--tag-output",
            "--mpi-global-args=--allow-run-as-root",
            "{PYTHON_INTERPRETER}",
            "$(location {})".format(main),
        ] + args,
        data = [
            "@local_config_poplar//poplar:mpirun",
            "@local_config_poplar//poplar:poprun",
        ],
        **kwargs
    )
