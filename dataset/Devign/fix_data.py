import json
import os

# 定义输入和输出文件路径
# 因为脚本在 Devign 目录下运行，所以文件名直接写文件名即可
input_train = 'train.jsonl'
output_train = 'train_fixed.jsonl'

input_valid = 'valid.jsonl'
output_valid = 'valid_fixed.jsonl'

input_test = 'test.jsonl'
output_test = 'test_fixed.jsonl'

def convert_json_array_to_jsonl(in_path, out_path):
    if not os.path.exists(in_path):
        print(f"⚠️  文件不存在: {in_path}，跳过。")
        return

    print(f"🔄 正在转换: {in_path} -> {out_path}")
    
    try:
        with open(in_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 处理空文件情况
        if not content:
            print(f"⚠️  文件为空: {in_path}")
            return

        # 去除首尾的方括号 (如果存在)
        if content.startswith('[') and content.endswith(']'):
            content = content[1:-1]
        
        # 解析整个列表
        # 注意：如果文件非常大（几百MB以上），这种一次性读取可能会占用较多内存
        # 对于 Devign 数据集通常没问题
        data_list = json.loads('[' + content + ']')
        
        with open(out_path, 'w', encoding='utf-8') as f_out:
            for item in data_list:
                # 写入标准的 JSONL 格式 (每行一个 JSON，无逗号，无外层括号)
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ 转换成功！共写入 {len(data_list)} 条数据到 {out_path}")

    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        print("提示：请确认原文件确实是 JSON 数组格式 [...]")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")

# ============ 主程序 ============
if __name__ == "__main__":
    print("开始转换 Devign 数据集格式 (JSON Array -> JSONL)...")
    print("-" * 30)
    
    convert_json_array_to_jsonl(input_train, output_train)
    convert_json_array_to_jsonl(input_valid, output_valid)
    convert_json_array_to_jsonl(input_test, output_test)
    
    print("-" * 30)
    print("所有任务完成！请检查生成的 *_fixed.jsonl 文件。")
    print("\n下一步操作:")
    print("运行训练脚本时，请将数据路径指向新生成的文件，例如:")
    print(f"  --train_data_file ./train_fixed.jsonl")
    print(f"  --eval_data_file ./valid_fixed.jsonl")
    print(f"  --test_data_file ./test_fixed.jsonl")