import os

def create_directories(Rmax, N, template_file):
    os.makedirs("Rk", exist_ok=True)

    if not os.path.exists(template_file):
        print(f"Error: Template file '{template_file}' not found.")
        return

    with open(template_file, "r") as f:
        template_content = f.read()

    for i in range(1, N + 1):
        sub_dir = os.path.join("Rk", f"R{i}")
        os.makedirs(sub_dir, exist_ok=True)

        Ri = i * Rmax / N
        modified_content = template_content.replace("REPLACE", f"{Ri:.11f}")

        with open(os.path.join(sub_dir, os.path.basename(template_file)), "w") as f:
            f.write(modified_content)

    return None


