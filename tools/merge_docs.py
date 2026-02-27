import os
import re

def merge_experiments():
    base_dir = "实验记录"
    files = [
        "1.Data Analysis (前期)/实验1_论文初稿.md",
        "2.统计显著性检验/实验2_论文初稿.md",
        "3.消融实验/实验3_论文初稿.md",
        "4.Model Diagnostics (深度验证)/实验4_论文初稿.md"
    ]
    
    output_file = os.path.join(base_dir, "预测区间实验全文.md")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("# 预测区间实验全文报告\n\n")
        
        for idx, rel_path in enumerate(files, 1):
            full_path = os.path.join(base_dir, rel_path)
            if not os.path.exists(full_path):
                print(f"Warning: {full_path} not found.")
                continue
                
            with open(full_path, 'r', encoding='utf-8') as infile:
                content = infile.read()
                
                # 调整图片路径，因为合并后的文件在“实验记录/”下
                # 原来的图片路径是相对于各子目录的，例如：最终选择图像/1_分布特征.png
                # 需要改为：1.Data Analysis (前期)/最终选择图像/1_分布特征.png
                # 注意处理 URL 编码的情况（虽然 md 中可能是中文，但 pandoc 或 python 处理时可能涉及）
                sub_dir = os.path.dirname(rel_path)
                if sub_dir:
                    # 匹配 ![alt](path)
                    def fix_path(match):
                        alt = match.group(1)
                        path = match.group(2)
                        # 如果已经是绝对路径或 http 开头，不处理
                        if path.startswith(('http', '/', 'C:', 'D:')):
                            return f'![{alt}]({path})'
                        # 拼接子目录
                        new_path = f"{sub_dir}/{path}"
                        return f'![{alt}]({new_path})'
                    
                    content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', fix_path, content)
                
                outfile.write(f"\n---\n\n")
                outfile.write(f"## 实验 {idx}\n\n")
                outfile.write(content)
                outfile.write("\n\n")
                
    print(f"Merged to {output_file}")

if __name__ == "__main__":
    merge_experiments()
