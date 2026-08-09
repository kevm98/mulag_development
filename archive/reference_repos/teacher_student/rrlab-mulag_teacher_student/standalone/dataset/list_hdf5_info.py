# import h5py

# def print_h5_tree(name, obj):
#     if isinstance(obj, h5py.Dataset):
#         print(f"[DATASET] {name}")
#         print(f"  shape: {obj.shape}")
#         print(f"  dtype: {obj.dtype}")
#         print(f"  nbytes: {obj.nbytes}")
#     elif isinstance(obj, h5py.Group):
#         print(f"[GROUP]   {name}")

# h5_path = "datasets/annotated_dataset.hdf5"

# with h5py.File(h5_path, "r") as f:
#     f.visititems(print_h5_tree)


import h5py

h5_path = "/home/qili/Software/IsaacSim_Exts/rrlab/logs/mulag_eval/experiment_info/mulag_experiment_info.hdf5"

with h5py.File(h5_path, "r") as f:
    data_group = f["data"]

    # ---- total demos ----
    demo_keys = list(data_group.keys())
    num_demos = len(demo_keys)
    print(f"\nTotal demos: {num_demos}")

    # ---- pick one demo (first) ----
    demo_name = demo_keys[0]
    demo = data_group[demo_name]

    print(f"\nExample demo: {demo_name}")

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name} -> shape={obj.shape}, dtype={obj.dtype}, nbytes={obj.nbytes}")

    demo.visititems(visitor)

    # # ---- total transitions ----
    # total_steps = 0
    # for k in demo_keys:
    #     total_steps += data_group[k]["actions"].shape[0]

    # print(f"\nTotal transitions (sum over demos): {total_steps}")
