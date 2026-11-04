import json
import os
import glob
import re

def calculate_detailed_rewards(base_path):
    total_steps = 0
    reward_exact_1 = 0
    reward_positive = 0 
    file_count = 0

    json_files = glob.glob(os.path.join(base_path, "**/*.json"), recursive=True)
    
    if not json_files:
        print(f"エラー: {base_path} 内にファイルが見つかりません。")
        return

    def natural_keys(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]

    # 1. 昇順にソート（古い順から新しい順へ）
    json_files.sort(key=natural_keys)

    # 2. 【重要】最新の50個だけを抽出
    json_files = json_files[-50:]

    print(f"--- 最新の {len(json_files)} ファイルを集計します ---")
    print(f"{'File Name':<30} | {'Reward':<10}")
    print("-" * 45)

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                filename = os.path.basename(file_path)
                
                entries = data if isinstance(data, list) else [data]
                
                for entry in entries:
                    reward = entry.get("reward")
                    if reward is not None:
                        total_steps += 1
                        if reward == 1 or reward == 1.0:
                            reward_exact_1 += 1
                        
                        if reward > 0:
                            reward_positive += 1
                        
                        print(f"{filename:<30} | {reward:<10}")
                file_count += 1
                    
        except Exception as e:
            # エラーがあった場合にスキップするだけでなく、原因を把握したい場合は print(e) を追加
            pass

    if total_steps > 0:
        print("\n" + "="*40)
        print(f"【最新50ファイル 集計結果】")
        print(f"解析ファイル総数 : {file_count}")
        print(f"総ステップ数     : {total_steps}")
        print("-" * 40)
        print(f"正答率 (Reward == 1.0): {reward_exact_1} 件 ({(reward_exact_1/total_steps)*100:.1f}%)")
        print(f"正の報酬率 (Reward > 0): {reward_positive} 件 ({(reward_positive/total_steps)*100:.1f}%)")
        print("="*40)
    else:
        print("\n有効なデータが見つかりませんでした。")

if __name__ == "__main__":
    # ここは実際のディレクトリに合わせて適宜変更してください
    target_dir = "data/food_task/14/rl/logs/train"
    calculate_detailed_rewards(target_dir)