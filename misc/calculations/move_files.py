import os
import sys

if __name__ == "__main__":

    offset = int(sys.argv[2])
    for file in os.listdir(sys.argv[1]):
        if file[:22] == "Experiment_Explanation":
            original_file = f"{sys.argv[1]}/{file}"
            file = file.split("_")
            episode_num = int(file[2])
            episode_step = file[3]
            os.rename(
                original_file, f"./new_files/Experiment_Explanation_{episode_num + offset}_{episode_step}"
            )
        elif file[:10] == "action_mem":
            original_file = f"{sys.argv[1]}/{file}"
            file = file.split("_")
            episode_num = int(file[2].split(".")[0])
            os.rename(
                original_file, f"./new_files/action_mem_{episode_num + offset}.npy"
            )
        elif file[:14] == "ref_action_mem":
            original_file = f"{sys.argv[1]}/{file}"
            file = file.split("_")
            episode_num = int(file[3].split(".")[0])
            os.rename(
                original_file, f"./new_files/ref_action_mem_{episode_num + offset}.npy"
            )
        elif file[:12] == "ref_expl_mem":
            original_file = f"{sys.argv[1]}/{file}"
            file = file.split("_")
            episode_num = int(file[3].split(".")[0])
            os.rename(
                original_file, f"./new_files/ref_expl_mem_{episode_num + offset}.npy"
            )
        elif file[:11] == "ref_fid_mem":
            original_file = f"{sys.argv[1]}/{file}"
            file = file.split("_")
            episode_num = int(file[3].split(".")[0])
            os.rename(
                original_file, f"./new_files/ref_fid_mem_{episode_num + offset}.npy"
            )
        elif file[:13] == "zn_action_mem":
            original_file = f"{sys.argv[1]}/{file}"
            file = file.split("_")
            episode_num = int(file[3].split(".")[0])
            os.rename(
                original_file, f"./new_files/zn_action_mem_{episode_num + offset}.npy"
            )
        elif file[:11] == "zn_expl_mem":
            original_file = f"{sys.argv[1]}/{file}"
            file = file.split("_")
            episode_num = int(file[3].split(".")[0])
            os.rename(
                original_file, f"./new_files/zn_expl_mem_{episode_num + offset}.npy"
            )
        elif file[:10] == "zn_fid_mem":
            original_file = f"{sys.argv[1]}/{file}"
            file = file.split("_")
            episode_num = int(file[3].split(".")[0])
            os.rename(
                original_file, f"./new_files/zn_fid_mem_{episode_num + offset}.npy"
            )
        elif file[:13] == "zo_action_mem":
            original_file = f"{sys.argv[1]}/{file}"
            file = file.split("_")
            episode_num = int(file[3].split(".")[0])
            os.rename(
                original_file, f"./new_files/zo_action_mem_{episode_num + offset}.npy"
            )
        elif file[:11] == "zo_expl_mem":
            original_file = f"{sys.argv[1]}/{file}"
            file = file.split("_")
            episode_num = int(file[3].split(".")[0])
            os.rename(
                original_file, f"./new_files/zo_expl_mem_{episode_num + offset}.npy"
            )
        elif file[:10] == "zo_fid_mem":
            original_file = f"{sys.argv[1]}/{file}"
            file = file.split("_")
            episode_num = int(file[3].split(".")[0])
            os.rename(
                original_file, f"./new_files/zo_fid_mem_{episode_num + offset}.npy"
            )
        elif file[:18] == "Experiment_actions":
            original_file = f"{sys.argv[1]}/{file}"
            file = file.split("_")
            episode_num = int(file[2].split(".")[0])
            os.rename(
                original_file, f"./new_files/Experiment_actions_{episode_num + offset}.svg"
            )
        else:  # Experiment_x_fidelities
            original_file = f"{sys.argv[1]}/{file}"
            file = file.split("_")
            episode_num = int(file[1])
            os.rename(
                original_file, f"./new_files/Experiment_{episode_num + offset}_fidelities.svg"
            )
