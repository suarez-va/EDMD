import os
import shutil

def create_directories(Rmax, N, template_file):
    main_dir = "Rk"
    os.makedirs(main_dir, exist_ok=True)

    for i in range(1,N+1):
        sub_dir = os.path.join(main_dir, f"R{i}")
        os.makedirs(sub_dir, exist_ok=True)

        if os.path.exists(template_file):
            with open(template_file, "r") as f:
                content = f.read()

            Ri = i * Rmax / N
            modified_content = content.replace("REPLACE", str(Ri))

            new_file_path = os.path.join(sub_dir, os.path.basename(template_file))
            with open(new_file_path, "w") as f:
                f.write(modified_content)

    return None

