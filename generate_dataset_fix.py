import os
import json
from glob import glob
from tqdm import tqdm

def round2(val):
    try:
        return round(float(val), 2)
    except Exception:
        return val

def extract_max_deceleration(future_frames, max_limit=-10):
    max_decel = 0.0
    for frame in future_frames:
        chassis = frame.get("chassis")
        if chassis and len(chassis) > 0 and len(chassis[0]) > 1:
            accel = chassis[0][1]
            if accel < max_decel:
                max_decel = accel
    max_decel = max(max_decel, max_limit)
    return round2(max_decel)

def recommend_deceleration_from_future(future_frames, decel_threshold=-0.5, max_limit=-10):
    max_decel = extract_max_deceleration(future_frames, max_limit)
    if max_decel < decel_threshold:
        return f"Deceleration is needed, the recommended deceleration is {max_decel:.2f} m/s^2."
    else:
        return "No deceleration is needed."

def filter_and_sort_radar(radar_list):
    radar_indices = [
        (0, "longitudinal_dist", "m"),
        (1, "lateral_dist", "m"),
        (5, "orientation", "rad"),
        (7, "closest_long_dist", "m"),
        (8, "closest_lat_dist", "m"),
        (13, "rcs", "m^2"),
        (14, "confidence", ""),
        (16, "longitudinal_vel", "m/s"),
        (17, "lateral_vel", "m/s"),
    ]
    filtered = []
    for obj in radar_list:
        if isinstance(obj, list):
            d = {}
            for idx, name, unit in radar_indices:
                if idx < len(obj):
                    val = round2(obj[idx])
                    d[name] = f"{val}{unit}" if unit and name != "confidence" else f"{val}"
            if "confidence" in d and float(d.get("confidence", 0)) > 30:
                filtered.append(d)
        elif isinstance(obj, dict):
            d = {}
            for _, name, unit in radar_indices:
                if name in obj:
                    val = round2(obj[name])
                    d[name] = f"{val}{unit}" if unit and name != "confidence" else f"{val}"
            if "confidence" in d and float(d.get("confidence", 0)) > 30:
                filtered.append(d)
    filtered = sorted(filtered, key=lambda x: float(x.get("confidence", 0)), reverse=True)
    return filtered[:5] if filtered else []

def process_single_json(json_path, history_len=20, future_len=10):
    output_list = []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    frames = data["frames"] if isinstance(data, dict) and "frames" in data else data
    total = len(frames)
    idx = 0
    while True:
        if idx + history_len + future_len <= total:
            cur_frames = frames[idx: idx + history_len]
            future_frames = frames[idx + history_len: idx + history_len + future_len]
            idx += history_len
        else:
            if total - idx <= 1:  
                break
            cur_frames = frames[idx: total - 1]
            future_frames = [frames[total - 1]]
            idx = total  

        images = []
        for f in cur_frames:
            if "camera" in f and "calib_img_path" in f["camera"]:
                images.append(f["camera"]["calib_img_path"])
        n_imgs = len(images)

        last_frame = cur_frames[-1]
        last_radar = last_frame.get("radar", [])
        radar_targets = filter_and_sort_radar(last_radar) if last_radar else []
        radar_lines = []
        for i, r in enumerate(radar_targets):
            obstacle_label = f"obstacles{i+1}:"
            r_str = ", ".join([f"{k}: {v}" for k, v in r.items()])
            radar_lines.append(f"{obstacle_label}{r_str and ' ' + r_str or ''}")

        chassis_raw = last_frame.get("chassis", [[0,0,0,0,0]])
        chassis_lines = []
        if chassis_raw and len(chassis_raw[0]) >= 5:
            chassis_lines = [
                f"ego speed(km/h): {round2(chassis_raw[0][0])}",
                f"ego acceleration(m/s^2): {round2(chassis_raw[0][1])}",
                f"ego yaw_rate(rad/s): {round2(chassis_raw[0][3])}",
                f"ego steering_angle(rad): {round2(chassis_raw[0][4])}"
            ]

        images_tag = "<image>" * n_imgs
        header_text = (
            f"Suppose you are driving.\n"
            f"I will provide you with a description including key information that affects the deceleration decision,\n"
            f"such as {images_tag} ({n_imgs} historical camera images within 1 second), location and motion status of surrounding obstacles provided by radar,\n"
            f"and the ego vehicle's own motion status.\n"
            f"Based on all this information, tell me if deceleration is needed, and if so, what is the recommended deceleration value (unit: m/s^2, up to 10 m/s^2 at most)."
        )
        info_text = (
            "Suppose you are driving, here is the scene information I provide:\n"
            "ego vehicle status:\n" +
            ("\n".join(chassis_lines) if chassis_lines else "None") +
            "\nNext, I will provide the information of surrounding obstacles:\n" +
            ("\n".join(radar_lines) if radar_lines else "None")
        )
        user_content = f"{header_text}\n{info_text}"
        assistant_content = recommend_deceleration_from_future(future_frames, max_limit=-10)
        entry = {
            "messages": [
                {
                    "content": user_content,
                    "role": "user"
                },
                {
                    "content": assistant_content,
                    "role": "assistant"
                }
            ],
            "images": images
        }
        output_list.append(entry)
        if idx >= total:
            break
    return output_list

def process_folder(input_folder, output_folder, file_pattern="*.json"):
    all_jsons = glob(os.path.join(input_folder, file_pattern))
    print(f"Found {len(all_jsons)} JSON files in {input_folder}.")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for json_file in tqdm(all_jsons, desc="Processing JSON files"):
        output_list = []
        try:
            output_list = process_single_json(json_file)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
        if output_list:
            base_name = os.path.splitext(os.path.basename(json_file))[0]
            out_path = os.path.join(output_folder, f"{base_name}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for item in output_list:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Wrote {len(output_list)} samples to {out_path}")

if __name__ == "__main__":
    input_folder = "/data1/coolwin.fu/code/AEB_datatool/train_data"     # 修改为你的数据目录
    output_folder = "/data1/coolwin.fu/docker/llama-factory/workspace/LLaMA-Factory/data/Qwen2.5_vl_LXJ/LXJ_dataset"  # 修改为你的输出目录
    process_folder(input_folder, output_folder)