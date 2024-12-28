import re
import subprocess


def extract_packages_from_poetry_lock(file_path, output_file):
    try:
        packages = []
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 使用正则表达式提取 package 名和 version
        package_pattern = r'\[\[package\]\]\s+name\s=\s"([^"]+)"\s+version\s=\s"([^"]+)"'
        matches = re.findall(package_pattern, content)

        # 将包名和版本存储为 requirements.txt 格式
        with open(output_file, 'w', encoding='utf-8') as out_file:
            for name, version in matches:
                out_file.write(f"{name}=={version}\n")
                packages.append(f"{name}=={version}")

        print(f"Requirements written to {output_file}")
        return packages

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return []



if __name__ == "__main__":
    poetry_lock_file = "poetry.lock"  # 替换为你的 poetry.lock 文件路径
    requirements_file = "requirements.txt"  # 输出文件名

    # 提取包并生成 requirements.txt
    extracted_packages = extract_packages_from_poetry_lock(poetry_lock_file, requirements_file)

    if extracted_packages:
        print("Packages to install:")
        for package in extracted_packages:
            print(package)
    else:
        print("No packages found or the file is empty.")
