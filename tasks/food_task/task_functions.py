import numpy as np

def reward_function(current_env, action_waveform, next_env) -> np.ndarray :
    # internal_state = current_env.internal_state

    # leftimage_rgb = np.mean(current_env.dialogue_scene.images[0], axis = (0, 1))
    # rightimage_rgb = np.mean(current_env.dialogue_scene.images[1], axis = (0, 1))
    # leftimage_distance = np.linalg.norm(leftimage_rgb - internal_state)
    # rightimage_distance = np.linalg.norm(rightimage_rgb - internal_state)

    # 内部状態に近い方を正解画像とする
    # if leftimage_distance < rightimage_distance:
    # correct_image = current_env.dialogue_scene.images[0]
    # else:
    #     correct_image = current_env.dialogue_scene.images[1]

    # 環境側で選択されたアイテムが正解画像と一致しているか確認
    # item_correct = (next_env.dialogue_feedback.selected_item is not None and 
                    # np.array_equal(next_env.dialogue_feedback.selected_item.image, correct_image))

    # --- 2. LLMによる質問への回答品質の判定 ---
    # dialogue_simulator.pyでLLMから取得し、feedbackに格納した response_quality を参照
    response_quality = getattr(next_env.dialogue_feedback, "response_quality", 0)

    # --- 3. 最終報酬の算出 ---
    # アイテムが正解であり、かつLLMが「質問に対して適切な回答」と判定した場合に最大報酬
    if response_quality == 1:
        reward = 1.0
    # # アイテムは合っているが、答え方が不十分な場合は部分報酬（学習を促進するため）
    # elif item_correct:
    #     reward = 0.5
    # それ以外（アイテム間違い、または意味不明な回答）は報酬ゼロ
    else:
        reward = -1.0

    return reward

def termination_function(current_env, next_env):
    if next_env.dialogue_scene.scene_id == 1:
        terminated = True
    else:
        terminated = False
    return terminated

def truncation_function(next_env):
    truncated = True
    return truncated

def internal_state_update_function(internal_state, dialogue_feedback):
    # next_internal_state = np.random.random(3).astype(np.float32)*255
    next_internal_state = internal_state
    return next_internal_state

def observation_function(next_env):
    observation = {}
    observation['state'] = next_env.internal_state
    
    # 画像の取得（既存のまま）
    if len(next_env.dialogue_scene.images) >= 1:
        observation['leftimage'] = next_env.dialogue_scene.images[0]
    else:
        observation['leftimage'] = np.zeros((224, 224, 3))

    # 質問音声はステップごとに対話シミュレータが合成する（卒論4.1）
    observation['audio'] = next_env.dialogue_scene.prompt_waveform

    return observation

def initial_internal_state_function():
    initial_internal_state = np.random.random(3).astype(np.float32)*255
    return initial_internal_state
