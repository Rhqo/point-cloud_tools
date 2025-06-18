from collections import defaultdict

def analyze_dataset_files(file_path):
    class_counts = defaultdict(int)

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.replace('\\', '/').split('/')
                if len(parts) > 1:
                    class_name = parts[-2]
                    class_counts[class_name] += 1
    
    total_files = sum(class_counts.values())

    print(f"Total: {total_files}")
    
    for class_name, count in sorted(class_counts.items()):
        print(f"- {class_name}: {count}")

dataset_file_path = "./Crops3D/test.txt"
analyze_dataset_files(dataset_file_path)
