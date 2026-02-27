import tkinter as tk
from tkinter import messagebox, filedialog
import os

class PyCopierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Python代码合并复制工具")
        self.root.geometry("400x500")

        # 获取当前脚本所在目录
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 顶部标签
        self.lbl_path = tk.Label(root, text=f"当前路径:\n{self.current_dir}", wraplength=380, fg="gray")
        self.lbl_path.pack(pady=5)

        # 列表框框架 (包含滚动条)
        self.frame_list = tk.Frame(root)
        self.frame_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.scrollbar = tk.Scrollbar(self.frame_list)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Listbox: selectmode=MULTIPLE 允许点击即选中/取消
        self.file_listbox = tk.Listbox(
            self.frame_list, 
            selectmode=tk.MULTIPLE, 
            yscrollcommand=self.scrollbar.set,
            font=("Consolas", 10)
        )
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.file_listbox.yview)

        # 底部按钮区域
        self.frame_btns = tk.Frame(root)
        self.frame_btns.pack(fill=tk.X, padx=10, pady=10)

        self.btn_select_all = tk.Button(self.frame_btns, text="全选", command=self.select_all)
        self.btn_select_all.pack(side=tk.LEFT, padx=5)

        self.btn_refresh = tk.Button(self.frame_btns, text="刷新列表", command=self.load_files)
        self.btn_refresh.pack(side=tk.LEFT, padx=5)

        self.btn_copy = tk.Button(
            self.frame_btns, 
            text="复制选中内容到剪贴板", 
            command=self.copy_to_clipboard,
            bg="#007bff", fg="white", font=("Arial", 10, "bold")
        )
        self.btn_copy.pack(side=tk.RIGHT, padx=5)

        # 初始化加载文件
        self.load_files()

    def load_files(self):
        """扫描当前目录下的py文件并填充到列表"""
        self.file_listbox.delete(0, tk.END)
        try:
            files = [f for f in os.listdir(self.current_dir) if f.endswith('.py')]
            # 排序，保持列表整洁
            files.sort()
            
            if not files:
                self.file_listbox.insert(tk.END, "当前目录下无 .py 文件")
                self.file_listbox.config(state=tk.DISABLED)
                return

            self.file_listbox.config(state=tk.NORMAL)
            for f in files:
                self.file_listbox.insert(tk.END, f)
        except Exception as e:
            messagebox.showerror("错误", f"读取目录失败: {str(e)}")

    def select_all(self):
        """全选列表中的所有项"""
        if self.file_listbox['state'] == tk.NORMAL:
            self.file_listbox.select_set(0, tk.END)

    def copy_to_clipboard(self):
        """读取选中文件并复制到剪贴板"""
        selected_indices = self.file_listbox.curselection()
        
        if not selected_indices:
            messagebox.showwarning("提示", "请至少选择一个文件！")
            return

        final_content = []
        success_count = 0
        
        for index in selected_indices:
            filename = self.file_listbox.get(index)
            filepath = os.path.join(self.current_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 拼接格式：文件名 + 内容
                separator = "=" * 40
                file_block = f"{separator}\nFilename: {filename}\n{separator}\n\n{content}\n\n"
                final_content.append(file_block)
                success_count += 1
            except Exception as e:
                # 如果某个文件读取失败（例如编码问题），将其记录在输出中但不要崩溃
                error_msg = f"Error reading {filename}: {str(e)}\n\n"
                final_content.append(error_msg)

        # 将内容写入剪贴板
        full_text = "".join(final_content)
        self.root.clipboard_clear()
        self.root.clipboard_append(full_text)
        self.root.update() # 保持剪贴板更新

        messagebox.showinfo("成功", f"已复制 {success_count} 个文件的内容到剪贴板！")

if __name__ == "__main__":
    root = tk.Tk()
    app = PyCopierApp(root)
    root.mainloop()