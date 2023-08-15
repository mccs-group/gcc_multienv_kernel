filegroup(
	name = "gcc-multienv-kernel-files",
	srcs = [
		"bench_kernel.py",
	],
    visibility = ["//visibility:public"],
)

genrule(
    name = "gcc-multienv-kernel-bin",
    srcs = [
        ":gcc-multienv-kernel-files",
    ],
    outs = [
        "gcc-multienv-kernel",
    ],
    cmd = "cp $(location :gcc-multienv-kernel-files) $@ && " +
        "chmod 777 $@",
    visibility = ["//visibility:public"],
)

